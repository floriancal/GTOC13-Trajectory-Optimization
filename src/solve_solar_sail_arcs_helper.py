# Import needed modules
import numpy as np
import csv
from collections import defaultdict
import pykep as pk
from distlink import COrbitData, MOID_fast
import orbital
from scipy.integrate import solve_ivp
import re
from scipy.interpolate import PchipInterpolator
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from numba import njit
from scipy.interpolate import RegularGridInterpolator
import os
import shutil
from datetime import datetime
import threading
import random
from constants import MU_ALTAIRA, AU, t_max, C, A, m, r0


# ---- Solar Sail filter ----
def sail_filter_fast(
    r1,
    r2,
    t,
    vin_fb,
    v_init_lmbrt,
    C,
    r0,
    A,
    m,
    mu,
    planet,
    vp,
    gamma=np.deg2rad(90.0),
    Nsamples=6,
    GA_samples=100,
    Anomaly_samples=20,
    dv_margin_coef=5,
):
    """
    Fast feasibility filter for a Lambert arc using a solar sail.

    This function evaluates whether a Lambert transfer arc (from r1 to r2
    over time t) is likely to be achievable by a solar sail spacecraft,
    taking into account the velocity change provided by a preceding flyby,
    sail acceleration limits, and geometric efficiency constraints.

    The filter is conservative and intended for use in large-scale
    trajectory searches (e.g. GA / tree search), where unfeasible
    candidates must be rejected quickly.

    Parameters
    ----------
    r1 : callable
        Function of time returning the departure position vector (m).
    r2 : callable
        Function of time returning the arrival position vector (m).
    t : float
        Transfer time of the Lambert arc (s).
    vin_fb : array_like, shape (3,)
        Incoming spacecraft velocity before the flyby preceding
        the current Lambert arc (m/s).
    v_init_lmbrt : array_like, shape (3,)
        Departure velocity required by the Lambert solution (m/s).
    C : float
        Solar radiation pressure coefficient (Altaira flux) at 1 AU.
    r0 : float
        Reference distance r0 (same unit as r1 and r2, typically km or m).
    A : float
        Solar sail area (m²).
    m : float
        Spacecraft mass (kg).
    mu : float
        Gravitational parameter of the central body.
    planet : pykep.planet
        Planet object used for the flyby.
    vp : array_like, shape (3,)
        Planet heliocentric velocity vector at flyby epoch.
    gamma : float, optional
        Half-angle of the admissible sail orientation cone (rad).
        Default is 90 degrees.
    Nsamples : int, optional
        Number of samples along the Lambert arc.
    GA_samples : int, optional
        Number of flyby-provided delta-v samples to test.
    Anomaly_samples : int, optional
        Number of anomaly samples used to evaluate maneuver efficiency.
    dv_margin_coef : float, optional
        Conservative margin coefficient applied to delta-v feasibility.

    Returns
    -------
    feasible : bool
        True if the Lambert arc is deemed feasible with the solar sail.
    dv_left : ndarray, shape (3,)
        Remaining delta-v vector to be provided by the sail (m/s).
        Returns a zero vector if infeasible.
    vout : ndarray, shape (3,)
        Outgoing heliocentric velocity after flyby correction (m/s).
        Returns a zero vector if infeasible.

    Notes
    -----
    - Flyby constraints are evaluated using pykep flyby models.
    - Sail acceleration capability is approximated using the mean
      acceleration between departure and arrival distances.
    - Efficiency of the sail maneuver is estimated via the alignment
      between the remaining delta-v direction and the position vector.
    - Multiple conservative checks are applied to reject marginal or
      poorly conditioned solutions.

    This function is intended as a fast pre-filter AND DOES OT GUARANTEE THAT THE ARC IS FEASIBLE.
    """

    # 0) Conversion to planet frame
    vin_fb = np.array(vin_fb) - np.array(vp)
    v_init_lmbrt = np.array(v_init_lmbrt) - np.array(vp)

    # 1) delta-v requis
    dv_req = pk.fb_vel(vin_fb, v_init_lmbrt, planet)
    v2_eq, delta_ineq = pk.fb_con(vin_fb, v_init_lmbrt, planet)

    # if dv_req < 0.01:
    #    return True,[0,0,0], [0,0,0]

    # compute DV max givable by solar sail
    # we take the mean at distance departure and arrival
    a_vals1 = -2 * C * A / m * (r0 / norm(r1)) ** 2  # m/s^2
    a_vals2 = -2 * C * A / m * (r0 / norm(r2)) ** 2  # m/s^2
    a_vals = (a_vals1 + a_vals2) / 2
    dv_opt = norm(a_vals) * t

    if dv_opt < dv_req * dv_margin_coef:
        return False, [0, 0, 0], [0, 0, 0]

    # Integrate the contribution of the GA from the previous planet

    # First we check the violation of the constraints
    v2_eq, delta_ineq = pk.fb_con(vin_fb, v_init_lmbrt, planet)

    if np.isnan(delta_ineq):
        delta_ineq = 0
 
    # From this we can deduce the actual vout direction and magnitude if any different from requested
    # direction is satisfied
    if delta_ineq <= 0:
        vout = v_init_lmbrt / np.linalg.norm(v_init_lmbrt) * np.linalg.norm(vin_fb)

    # Direction is not satisfied
    else:
        # we obtain the output vector after planet
        vrot = rotate_towards(v_init_lmbrt, vin_fb, delta_ineq)
        vout = vrot / np.linalg.norm(vrot) * np.linalg.norm(vin_fb)

    dv_left = v_init_lmbrt - vout
    # retour repère heliocentrique
    vout = vout + np.array(vp)

    # l'arc est il naturellement faisable en conique ?
    if t < 0.001:
        t = 0.001
    rt, vt = pk.propagate_lagrangian(r0=r1, v0=vout, tof=t, mu=MU_ALTAIRA)
    delta = np.linalg.norm(np.array(r2) - np.array(r1))
    if delta < 100:
        print("conic arc found")
        return True, [0, 0, 0], vout

    # On verifie que le dv a fournir est dans la bonne demi-sphère alpha beta
    [alpha0, beta0] = alpha_beta_from_u(r1, vout, dv_left, eps=1e-12)
    if alpha0 > np.pi / 2 or alpha0 < -np.pi / 2:
        return False, [0, 0, 0], [0, 0, 0]
    if beta0 > np.pi / 2 or beta0 < -np.pi / 2:
        return False, [0, 0, 0], [0, 0, 0]

    # Obtain osculating elements
    el = list(pk.ic2par(r1, vout, mu))

    # Compute orbital period
    orb_t = 2 * np.pi * np.sqrt(el[0] ** 3 / mu)
    mean_mov = np.sqrt(mu / el[0] ** 3)
    N_rev = t / orb_t
    if N_rev > 1 and el[1] < 1:
        M = np.linspace(0, 2 * np.pi, Anomaly_samples)
    else:
        # Convert Ecc. ano. to Mean ano.
        # Mi = orbital.utilities.mean_anomaly_from_eccentric(el[1], el[5])
        Mi = mean_anomaly_from_ecc(el[1], el[5])
        Mf = Mi + mean_mov * t
        M = np.linspace(Mi, Mf, Anomaly_samples)

    eff = np.empty(Anomaly_samples)

    for i in range(Anomaly_samples):
        m_c = M[i]
        # Convert mean. ano. --> Ecc. ano.
        # m_c_ecc = orbital.utilities.eccentric_anomaly_from_mean(el[1], m_c)
        # m_c_ecc = anomaly_from_mean(el[1], m_c)
        m_c_ecc = kepler_anomaly_from_mean(el[1], m_c)
        el[5] = m_c_ecc
        ta = true_anomaly_from_anomaly(el[1], el[5])
        r, v = pk.par2ic(el, mu)
        dv_left_inertial = dv_left

        # get the the orbital change efficiency
        eff[i] = cos_angle_between(r, dv_left_inertial) ** 2

        # include an efficiency wrt orbital position for planar change TBD

    mean_eff = np.mean(eff)

    # check transfer acceptability
    # margin defined in inputs is abosrbing non modeled parameters (approximation, oberth effect)

    # we add also a contribution for phasing delay
    vmax = calc_vmax_no_decel(0, norm(dv_left), abs(a_vals), t)
    if np.isnan(vmax):
        return False, [0, 0, 0], [0, 0, 0]
    v_phasing_add = vmax - norm(dv_left)

    covering_man = (norm(dv_left) + v_phasing_add) * dv_margin_coef - dv_opt * mean_eff
    if covering_man > 0:
        return False, [0, 0, 0], [0, 0, 0]

    # If here manoeuver is supposely feasible ! Compute reward with another function
    return True, dv_left, vout

def angle_between(a, b):
    """
    Compute the angle between two vectors.

    The angle is computed from the normalized dot product and returned
    in radians. The cosine of the angle is clamped to the interval
    [-1, 1] to avoid numerical errors due to floating-point roundoff.

    Parameters
    ----------
    a : array_like
        First input vector.
    b : array_like
        Second input vector.

    Returns
    -------
    theta : float
        Angle between vectors `a` and `b`, in radians.

    Notes
    -----
    - If either vector has zero norm, the result is undefined.
    """
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    cos_theta = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    # éviter les erreurs numériques (ex: cos > 1 à cause de roundoff)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.arccos(cos_theta)  # en radians
	
	
	
	def u_from_alpha_beta(r, v, alpha, beta, eps=1e-12):
    """
    Convert sail angles (alpha, beta) back to the inertial unit vector `u`
    using the same local RTN-like basis built from position r and velocity v.

    Inputs:
      r : array_like (3,) position vector
      v : array_like (3,) velocity vector
      alpha : float (radians)
      beta  : float (radians)
      eps : small number to detect singularities

    Returns:
      u : ndarray (3,), inertial unit vector
    """
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)

    rnorm = np.linalg.norm(r)
    if rnorm < eps:
        raise ValueError("r too small")
    rhat = r / rnorm

    h = np.cross(r, v)
    hnorm = np.linalg.norm(h)
    if hnorm < eps:
        # near-radial motion
        ez = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(ez, rhat)) > 0.99:
            ez = np.array([0.0, 1.0, 0.0])
        hhat = np.cross(rhat, ez)
        hhat /= np.linalg.norm(hhat)
    else:
        hhat = h / hnorm

    that = np.cross(hhat, rhat)

    # combinaison linéaire dans la base locale
    u = (
        np.cos(alpha) * rhat
        + np.sin(alpha) * np.cos(beta) * that
        + np.sin(alpha) * np.sin(beta) * hhat
    )

    # normalisation pour robustesse numérique
    return u / np.linalg.norm(u)


def alpha_beta_from_u(r, v, u, eps=1e-12):
    """
    Convert an inertial unit vector `u` (target sail-normal direction)
    to sail angles (alpha, beta) using the local RTN-like basis built
    from position r and velocity v.

    Inputs:
      r : array_like (3,) position vector (in same units as v)
      v : array_like (3,) velocity vector
      u : array_like (3,) target direction (doesn't need to be unit; will be normalized)
      eps : small number to detect singularities

    Returns:
      alpha : float (radians), in [0, pi]
      beta  : float (radians), in (-pi, pi]
    """
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)
    u = np.asarray(u, dtype=float)

    if np.linalg.norm(u) < eps:
        return 0, 0

    # normalize u
    u_hat = u / np.linalg.norm(u)

    # build local frame
    rnorm = np.linalg.norm(r)
    if rnorm < eps:
        raise ValueError("r too small")

    rhat = r / rnorm
    h = np.cross(r, v)
    hnorm = np.linalg.norm(h)
    if hnorm < eps:
        # near-radial motion: choose an arbitrary orthonormal basis perpendicular to rhat
        # pick a stable vector not parallel to rhat, e.g. e_z
        ez = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(ez, rhat)) > 0.99:
            ez = np.array([0.0, 1.0, 0.0])
        hhat = np.cross(rhat, ez)
        hhat /= np.linalg.norm(hhat)
    else:
        hhat = h / hnorm
    that = np.cross(hhat, rhat)

    # components of u in (rhat, that, hhat)
    a = float(np.dot(u_hat, rhat))  # = cos(alpha)
    b = float(np.dot(u_hat, that))  # = sin(alpha)*cos(beta)
    c = float(np.dot(u_hat, hhat))  # = sin(alpha)*sin(beta)

    # clamp numerical noise
    a = np.clip(a, -1.0, 1.0)
    alpha = np.arccos(a)  # in [0, pi]

    sin_alpha = np.sqrt(max(0.0, 1.0 - a * a))

    if sin_alpha < 1e-12:
        # alpha ~ 0 or pi, beta is undefined (all directions with same rhat component)
        # choose beta = 0 by convention
        beta = 0.0
    else:
        # recover beta from b and c
        # b = sinα * cosβ, c = sinα * sinβ -> atan2(c, b)
        beta = np.arctan2(c / sin_alpha, b / sin_alpha)

    return alpha, beta


def make_control_splines(par, N, tof, time_vector=None):
    """
    Crée des splines PCHIP pour alpha et beta.

    par : vecteur 1D de longueur 2*N
          mode='angles' : [alpha_nodes, beta_nodes] en radians
          mode='unconstrained' : variables non contraintes mappées en angles
    N : nombre de noeuds
    tof : durée totale
    mode : 'angles' ou 'unconstrained'

    Retour : alpha_spline(t), beta_spline(t) pour t dans [0, tof]
    """
    if time_vector is None:
        par = np.asarray(par).ravel()

        assert par.size == 2 * N, "par doit contenir 2*N éléments"

        t_nodes = np.linspace(0.0, tof, N)

    # Custom time vector (non linspace)
    else:
        N = len(time_vector)
        t_nodes = time_vector

    alpha_nodes = par[:N].copy()
    beta_nodes = par[N:].copy()
    alpha_nodes = np.clip(alpha_nodes, -np.pi / 2, np.pi / 2)
    beta_nodes = np.clip(beta_nodes, -np.pi / 2, np.pi / 2)

    alpha_spline = PchipInterpolator(t_nodes, alpha_nodes)
    beta_spline = PchipInterpolator(t_nodes, beta_nodes)

    return alpha_spline, beta_spline


@njit(cache=True, fastmath=True)
def sail_accel_func(r, v, alpha, beta):
    rnorm = np.linalg.norm(r)
    rhat = r / rnorm
    n = sail_normal_from_alpha_beta(r, v, alpha, beta)
    ndotr = np.dot(n, rhat)

    return -2 * C * A / m * (r0 / rnorm) ** 2 * ndotr**2 * n  # 0*n


@njit(cache=True, fastmath=True)
def sail_normal_from_alpha_beta(r, v, alpha, beta):
    rhat = r / np.linalg.norm(r)
    h = np.cross(r, v)
    hhat = h / np.linalg.norm(h)
    that = np.cross(hhat, rhat)

    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta), np.sin(beta)

    n = ca * rhat + sa * (cb * that + sb * hhat)
    return n / np.linalg.norm(n)

    # --- Fonction d’interpolation rapide (linéaire) ---


@njit(cache=True, fastmath=True)
def interp_fast(t, t_array, y_array):
    # suppose t dans [0, tof]
    idx = np.searchsorted(t_array, t) - 1
    if idx < 0:
        return y_array[0]
    if idx >= len(t_array) - 1:
        return y_array[-1]
    t1, t2 = t_array[idx], t_array[idx + 1]
    y1, y2 = y_array[idx], y_array[idx + 1]
    return y1 + (y2 - y1) * (t - t1) / (t2 - t1)


# --- Dynamique principale ---
@njit(cache=True, fastmath=True)
def dynamics(t, state, mu, par, tof, alpha_tab, beta_tab, t_cmd):
    r = state[0:3]
    v = state[3:6]

    # évaluation rapide des commandes
    alpha = interp_fast(t, t_cmd, alpha_tab)
    beta = interp_fast(t, t_cmd, beta_tab)
    rnorm = np.linalg.norm(r)
    a_grav = -mu * r / rnorm**3
    a_sail = sail_accel_func(r, v, alpha, beta)
    a_tot = a_grav + a_sail
    return np.concatenate((v, a_tot))


def propagate(r0, v0, tof, MU, par, N, time_vector=None):

    # Conditions initiales
    y0 = np.concatenate((r0, v0))

    # Création des splines de commande
    alpha_spline, beta_spline = make_control_splines(
        par, N, tof, time_vector=time_vector
    )

    # --- Pré-échantillonnage rapide ---
    N_cmd = 2000  # 1000 points par unité de temps
    # nombre d’échantillons de commande (ajuste selon tof)
    t_cmd = np.linspace(0, tof, N_cmd)
    alpha_tab = alpha_spline(t_cmd)
    beta_tab = beta_spline(t_cmd)

    # Intégration
    sol = solve_ivp(
        dynamics,
        t_span=(0, tof),
        y0=y0,
        args=(MU, par, tof, alpha_tab, beta_tab, t_cmd),
        method="RK45",  # ou 'RK45', 'Radau', 'LSODA'
        rtol=1e-7,  # NE PAS ALLER MOINS PRECIS QUE 1e-7
        atol=1e-7,
        dense_output=True,  # permet d’interpoler entre les pas
    )

    # Résultats
    t = sol.t
    r = sol.y[0:3, :].T
    v = sol.y[3:6, :].T
    return sol, r, v


def error_lmbrt(par, r0, v0, r2, v2, tof, MU, N):
    t, r_traj, v_traj = propagate(r0, v0, tof, MU, par, N)
    r_end = r_traj[-1]
    v_end = v_traj[-1]

    # coût : somme des carrés (plus lisse qu'une somme simple)
    err_pos = np.dot(r_end - r2, r_end - r2) / 10**10
    err_vel = 0  # np.dot(v_end - v2, v_end - v2)

    cost = err_pos + err_vel
    return float(cost)


def propagate_sub_control(
    r0, v0, tof, MU, par, N, alpha_tab_old, beta_tab_old, old_tof, prop_fac
):

    # Conditions initiales
    y0 = np.concatenate((r0, v0))

    # Création des splines de commande
    alpha_spline, beta_spline = make_control_splines(par, N, tof)

    # --- Pré-échantillonnage rapide ---
    N_cmd = 2000
    # nombre d’échantillons de commande
    t_cmd = np.linspace(0, tof, N_cmd)
    alpha_tab = alpha_spline(t_cmd)
    beta_tab = beta_spline(t_cmd)

    t_old_cmd = np.linspace(0, old_tof, N_cmd)
    t_cmd_reset = np.linspace(old_tof * prop_fac, old_tof, N_cmd)

    alpha_add = np.interp(t_cmd_reset, t_old_cmd, alpha_tab_old)
    beta_add = np.interp(t_cmd_reset, t_old_cmd, beta_tab_old)

    # somme du controle initial et du controle additionnel
    alpha_tab = alpha_tab + alpha_add
    beta_tab = beta_tab + beta_add

    # Intégration
    sol = solve_ivp(
        dynamics,
        t_span=(0, tof),
        y0=y0,
        args=(MU, par, tof, alpha_tab, beta_tab, t_cmd),
        method="RK45",  # ou 'RK45', 'Radau', 'LSODA'
        rtol=1e-7,  # NE PAS ALLER MOINS PRECIS QUE 1e-7
        atol=1e-7,
        dense_output=True,  # permet d’interpoler entre les pas
    )

    # Résultats
    t = sol.t
    r = sol.y[0:3, :].T
    v = sol.y[3:6, :].T
    return sol, r, v


def error_lmbrt_sub_control(
    par, r0, v0, r2, v2, tof, MU, N, alpha_tab, beta_tab, old_tof, prop_fac
):
    sol, r_traj, v_traj = propagate_sub_control(
        r0, v0, tof, MU, par, N, alpha_tab, beta_tab, old_tof, prop_fac
    )
    r_end = r_traj[-1]
    v_end = v_traj[-1]

    # coût : somme des carrés (plus lisse qu'une somme simple)
    err_pos = np.dot(r_end - r2, r_end - r2) / 10**10
    err_vel = 0  # np.dot(v_end - v2, v_end - v2)

    cost = err_pos + err_vel

    return float(cost)
  
# Anomalies conversions   
def mean_anomaly_from_ecc(e, anomaly):
    """
    Returns the mean anomaly M (elliptic, parabolic, or hyperbolic)
    from the given eccentricity e and anomaly (E or H).

    Parameters
    ----------
    e : float
        Eccentricity
    anomaly : float
        Eccentric (for e<1) or hyperbolic anomaly (for e>1)

    Returns
    -------
    M : float
        Mean anomaly
    """
    if e < 1.0:
        return anomaly - e * np.sin(anomaly)
    elif np.isclose(e, 1.0):
        # Parabolic case: Barker's equation (approx.)
        D = np.tan(anomaly / 2)
        return D + D**3 / 3.0
    else:  # e > 1
        return e * np.sinh(anomaly) - anomaly



def kepler_anomaly_from_mean(e, M, tol=1e-12, max_iter=60):
    """
    Retourne l'anomalie (E pour e<1, H pour e>1, D pour e≈1)
    à partir du mean anomaly M.
    Utilise : initial guess (Mikkola / log), Newton/Halley, et fallback bisection.
    """
    # normalisation pour elliptique
    if e < 1.0:
        # ramène M dans [0, 2pi)
        M0 = np.mod(M, 2 * np.pi)

        # --- 1) initial guess : Mikkola (bonne approximation pour large plage d'e) ---
        def mikkola(e, M):
            # implémentation classique de Mikkola (approximative mais robuste)
            alpha = (1.0 - e) / (4.0 * e + 0.5)
            beta = 0.5 * M / (4.0 * e + 0.5)
            with np.errstate(all="ignore"):
                z = (beta + np.sign(beta) * np.sqrt(beta**2 + alpha**3)) ** (1 / 3)
                s = z - alpha / z
                E0 = M + e * (3 * s - 4 * s**3)
            return E0

        E = mikkola(e, M0)

        # --- 2) Newton iterations (use Halley if desired) ---
        for _ in range(max_iter):
            f = E - e * np.sin(E) - M0
            f1 = 1 - e * np.cos(E)
            # Halley correction (optional, faster)
            f2 = e * np.sin(E)
            # Halley step:
            denom = 2 * f1 * f1 - f * f2
            if denom != 0:
                dE = 2 * f * f1 / denom
            else:
                dE = -f / f1
            E = E - dE
            if abs(dE) < tol:
                return E

        # --- 3) fallback : bisection on [0,2pi] (f monotone increasing in E) ---
        a, b = 0.0, 2 * np.pi
        fa = a - e * np.sin(a) - M0
        fb = b - e * np.sin(b) - M0
        # étendre si nécessaire (rare)
        for _ in range(200):
            if fa * fb <= 0:
                break
            b += 2 * np.pi
            fb = b - e * np.sin(b) - M0
        # bisection
        for _ in range(100):
            m = 0.5 * (a + b)
            fm = m - e * np.sin(m) - M0
            if abs(fm) < 1e-12:
                return m
            if fa * fm <= 0:
                b, fb = m, fm
            else:
                a, fa = m, fm
        return 0.5 * (a + b)

    elif np.isclose(e, 1.0):
        # Parabolic: Barker approx (D = tan(nu/2))
        # M ~ D + D^3/3  => on peut approximer D ≈ (3M/2)^(1/3) initial
        D = (1.5 * M) ** (1 / 3) if M >= 0 else -((1.5 * (-M)) ** (1 / 3))
        return D

    else:
        # Hyperbolic: solve e*sinh(H) - H = M
        # good initial guess (Vallado-like)
        if M == 0:
            H = 0.0
        else:
            sign = 1 if M > 0 else -1
            # initial guess using log approximation
            H = np.log(2 * abs(M) / e + 1.8) * sign

        # Newton on hyperbolic eqn
        for _ in range(max_iter):
            f = e * np.sinh(H) - H - M
            f1 = e * np.cosh(H) - 1
            dH = -f / f1
            H = H + dH
            if abs(dH) < tol:
                return H

        # fallback: bracket and bisection on [low, high]
        # choose bounds that bracket the root
        low = -10.0 if M < 0 else 0.0
        high = 10.0 if M > 0 else 0.0
        # expand until bracket found
        for _ in range(100):
            f_low = e * np.sinh(low) - low - M
            f_high = e * np.sinh(high) - high - M
            if f_low * f_high <= 0:
                break
            low *= 2
            high *= 2
        # bisection
        for _ in range(200):
            mid = 0.5 * (low + high)
            fm = e * np.sinh(mid) - mid - M
            if abs(fm) < 1e-12:
                return mid
            if (e * np.sinh(low) - low - M) * fm <= 0:
                high = mid
            else:
                low = mid
        return 0.5 * (low + high)


def true_anomaly_from_anomaly(e, anomaly):
    """
    Compute the true anomaly ν from the eccentric anomaly (E) or hyperbolic anomaly (H),
    automatically handling elliptical, parabolic (≈1), and hyperbolic cases.

    Parameters
    ----------
    e : float
        Eccentricity
    anomaly : float
        Eccentric anomaly (E) if e<1, Hyperbolic anomaly (H) if e>1, or Barker parameter D if e≈1

    Returns
    -------
    nu : float
        True anomaly in radians
    """

    if e < 1.0:
        # Elliptical case
        cos_nu = (np.cos(anomaly) - e) / (1 - e * np.cos(anomaly))
        sin_nu = (np.sqrt(1 - e**2) * np.sin(anomaly)) / (1 - e * np.cos(anomaly))
        nu = np.arctan2(sin_nu, cos_nu)
        return nu

    elif np.isclose(e, 1.0):
        # Parabolic case (approximation)
        D = anomaly
        nu = 2 * np.arctan(D)
        return nu

    else:
        # Hyperbolic case
        cosh_H = np.cosh(anomaly)
        sinh_H = np.sinh(anomaly)
        cosh_term = (e - cosh_H) / (e * cosh_H - 1)
        sinh_term = (np.sqrt(e**2 - 1) * sinh_H) / (e * cosh_H - 1)
        nu = np.arctan2(sinh_term, cosh_term)
        return nu    