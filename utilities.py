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




# Physical constants of the problem
MU_ALTAIRA = 1.39348062043343e11 * 1e9  # [m^3/s^2] GM of Altaira
DEG2RAD = np.pi / 180
AU = 149597870691
C = 5.4026 * 10**-6  # N/m^2
A = 15.0  # m^2
m = 500.0  # kg
r0 = 149597870691  # m



def find_contact(tof, rf, vf, rf_min, MU):
    """Propagate a state by a given time of flight and compute its distance
    to a reference target position.

    Parameters
    ----------
    tof : float
        Time of flight for propagation.
    rf : array_like
        Initial position vector.
    vf : array_like
        Initial velocity vector.
    rf_min : array_like
        Reference minimum-distance position.
    MU : float
        Gravitational parameter of the central body.

    Returns
    -------
    float
        Euclidean distance between propagated position and rf_min.
    """

    tof = float(np.atleast_1d(tof)[0])
    rf, vf = pk.propagate_lagrangian(r0=rf, v0=vf, tof=tof, mu=MU)
    cost = np.linalg.norm(np.array(rf) - np.array(rf_min))
    return cost


# ------------------------------------------------------------
# GTOC13 – Building the complete catalog of bodies of the system Altaira
# ------------------------------------------------------------
def load_bodies_from_csv(filename, mu_default=0.0):
    """
    Load a GTOC13 CSV file (planets, asteroids, or comets) and return
    a dictionary {id: pykep.planet}.

    Parameters
    ----------
    filename : str
        Path to the GTOC13 CSV file.
    mu_default : float, optional
        Default gravitational parameter [m^3/s^2] used when a body
        does not specify a GM value in the file.

    Returns
    -------
    dict
        Dictionary mapping body IDs to `pykep.planet.keplerian` objects.
    """
    bodies = {}

    with open(filename, newline="", encoding="cp1252") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Skip empty lines
            if not any(row.values()):
                continue  
            else:
                 # Find the correct ID field depending on file type
                for key in ("#Planet ID", "#Asteroid ID", "# Comet ID"):
                    if key in row:
                        body_id = int(row[key])
                        break

                name = row.get("Name", f"Body_{body_id}")

                # Orbital parameters read and convert
                sma = float(row["Semi-Major Axis (km)"]) * 1e3  # [m]
                ecc = float(row["Eccentricity ()"])
                inc = float(row["Inclination (deg)"]) * DEG2RAD  # [rad]
                Omega = float(row["Longitude of the Ascending Node (deg)"]) * DEG2RAD
                omega = float(row["Argument of Periapsis (deg)"]) * DEG2RAD

                if body_id > 1000 and body_id < 2000:
                    M0 = float(row["Mean Anomaly at t=0"]) * DEG2RAD
                else:
                    M0 = float(row["Mean Anomaly at t=0 (deg)"]) * DEG2RAD

                # Physical parameters
                mu = float(row.get("GM (km3/s2)", mu_default)) * 1e9  # [m^3/s^2]
                radius = float(row.get("Radius (km)", 0.0)) * 1e3  # [m]

                # Create the pykep planet object
                bodies[body_id] = pk.planet.keplerian(
                    pk.epoch(0, "mjd2000"),
                    (sma, ecc, inc, Omega, omega, M0),
                    MU_ALTAIRA,  # Central GM (Altaira)
                    mu,  # Body GM
                    radius,  # Equatorial radius
                    radius * 1.1,  # Safe radius (min flyby altitude)
                    name,
                )

    return bodies
    

def read_init_pos(filename="init_pos.csv"):
    """
    Read the CSV file 'init_pos.csv' and return several lists of data.

    Parameters
    ----------
    filename : str, optional
        Path to the CSV file (default is 'init_pos.csv').

    Returns
    -------
    shield_burned : bool
        Flag indicating whether the shield was burned (default False).
    V0_list : list of float
        List of initial velocities along x-axis.
    VFMIN_list : list of list of float
        List of reference velocities at body contact [vx, vy, vz].
    L_list : list of float
        List of L body parameter values.
    tp_list : list of float
        List of times at body arrival.
    pos_list : list of list of float
        List of initial positions [x, y, z].
    """
    
    shield_burned = False
    V0_list = []
    VFMIN_list = []
    L_list = []
    tp_list = []
    pos_list = []

    with open(filename, mode="r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)  # skip header
        rows = list(reader)  # materialize

        for values in rows:

            # --- Correction if last column lacks its comma due to a bug in the file generation ---
            if len(values) == 17:
                parts = values[16].split()
                if len(parts) == 2:
                    values[16] = parts[0]  # cost
                    values.append(parts[1])  # lambda0
                else:
                    raise ValueError(f"Ligne mal formée: {values}")

            # Check consistency of comumn number 
            if len(values) != 18:
                raise ValueError(
                    f"Nombre de colonnes inattendu: {len(values)} dans {values}"
                )

            # Get values
            iter_num = int(values[0])
            t0_opt = float(values[1])
            tp_first_pass_opt = float(values[2])
            pos_0_opt = [float(values[3]), float(values[4]), float(values[5])]
            v_0_opt = [float(values[6]), float(values[7]), float(values[8])]
            rf_min_opt = [float(values[9]), float(values[10]), float(values[11])]
            vf_min_opt = [float(values[12]), float(values[13]), float(values[14])]
            body_aimed = int(float(values[15]))
            cost = float(values[16])
            Lpar = float(values[17])

            # Fill the lists
            V0_list.append(v_0_opt[0])
            VFMIN_list.append(vf_min_opt)
            L_list.append(Lpar)
            tp_list.append(tp_first_pass_opt)
            pos_list.append(pos_0_opt)

    return shield_burned, V0_list, VFMIN_list, L_list, tp_list, pos_list



def time_find(par, V0, rf_min, vf_min, rf2_min, vf2_min, el_p, planet_init):
    """
    Determine the reference epoch and time-of-flight associated with a two-leg
    trajectory involving an initial spacecraft propagation and a planetary
    encounter, by numerically searching for contact times.

    This function computes:
      1) The time-of-flight from an initial spacecraft state to a first target
         position (first leg).
      2) The absolute time at which the target planet reaches the impact point.
      3) The corresponding initial epoch offset such that both events are
         time-consistent.

    The time-of-flight searches are performed using a brute-force numerical
    minimization (Nelder–Mead) of a contact residual function.

    Parameters
    ----------
    par : array-like
        Array of optimization parameters. The first element is used
        to define the initial spacecraft position along the y-axis.
    V0 : float
        Initial spacecraft velocity magnitude along the x-axis (m/s).
    rf_min : array-like
        Target position vector (m) for the first leg contact.
    vf_min : array-like
        Target velocity vector (m/s) for the first leg contact.
    rf2_min : array-like
        Target planet position vector (m) at the impact/contact point.
    vf2_min : array-like
        Target planet velocity vector (m/s) at the impact/contact point.
    el_p : array-like
        Classical orbital elements of the planet at the reference epoch.
        Expected ordering: [a, e, i, RAAN, ω, E]
    planet_init : object
        Planet object

    Returns
    -------
    t0 : float
        Initial epoch offset (s), defined such that:
        t0 + tof_init = tp_first_pass.
    tp_first_pass : float
        Absolute time (s) at which the planet reaches the impact point.
    tof_init : float
        Time-of-flight (s) from the spacecraft initial state to the first
        contact point.

    Notes
    -----
    - The initial spacecraft state is constructed as:
        rf = [-200 * AU, par[0], 0]
        vf = [V0, 0, 0]
    - The time-of-flight is obtained by minimizing a contact residual using
      the Nelder–Mead method with tight tolerances.
    - Planetary timing is estimated analytically using mean anomaly propagation,
      then refined with a second numerical minimization.
    - Multiple planetary revolutions are handled by estimating the number of
      full orbits elapsed during the time-of-flight.
    """
    # obtain tof from initial point to first planet (first leg)
    rf = [-200 * AU, par[0], 0]
    vf = [V0, 0, 0]
   
    current_guess = 10 * 86400 * 365.25
    res_try = minimize(
        find_contact,
        current_guess,
        args=(rf, vf, rf_min, MU_ALTAIRA),
        method="Nelder-Mead",
        options={"maxiter": 100000, "xatol": 1e-6, "fatol": 1e-6},
    )
    tof_init = float(np.atleast_1d(res_try.x)[0])
    invalid = False
    if res_try.fun > 100:
        invalid = True
        print("error, no convergence of tof finding")
    else:
        print("tof_init finding residual is :", res_try.fun)
        # obtain planet anomaly at contact (rf2, vf2 is the planet)
        el_2 = pk.ic2par(rf2_min, vf2_min, MU_ALTAIRA)
        E_impact_pl = el_2[5]

        # anomalies moyennes (radians)
        M_impact_pl = orbital.utilities.mean_anomaly_from_eccentric(el_2[1], el_2[5])
        M_0_pl = orbital.utilities.mean_anomaly_from_eccentric(el_p[1], el_p[5])

        # mean movement (rad/s)
        mean_mov_pl = np.sqrt(MU_ALTAIRA / el_2[0] ** 3)

        # difference of mean anomalies modulo 2π
        dM = (M_impact_pl - M_0_pl) % (2 * np.pi)

        # More than 1 orbit tof > period
        period_pl = 2 * np.pi / mean_mov_pl
        n_orbits = int(np.floor(tof_init / period_pl))

        # time associated at impact
        tp_first_pass = (dM / mean_mov_pl) + n_orbits * period_pl

        # Precision loop
        rp0, vp0 = planet_init.eph(pk.epoch(0))
        current_guess = tp_first_pass
        res_try = minimize(
            find_contact,
            current_guess,
            args=(rp0, vp0, rf2_min, MU_ALTAIRA),
            method="Nelder-Mead",
            options={"maxiter": 100000, "xatol": 1e-6, "fatol": 1e-6},
        )
        tp_first_pass = float(np.atleast_1d(res_try.x)[0])
        invalid = False
        if res_try.fun > 100:
            print("error, no convergence of tof finding")
            invalid = True
        else:
            print("tof_init finding residual is :", res_try.fun)

        # Time at mission begin
        t0 = tp_first_pass - tof_init

        return t0, tp_first_pass, tof_init



def distance_vector_gen(par, V0, rf2, vf2, MU):
    """
    Compute the minimum distance between two Keplerian trajectories
    over a fixed time window, using a bisection-based minimization.

    The first object is initialized on the Z=0 plane at the x-axis  
    initial -200 AU pos and a Y position given by par[0] an
    initial velocity along the +x direction. Its state is first
    propagated to perigee. The second object is defined by its
    initial Cartesian state (position and velocity).

    The function searches for the time of closest approach between
    the two objects within a given time window after perigee.

    Parameters
    ----------
    par : array_like
        Parameter vector. Only `par[0]` is used and represents the
        initial y-offset (impact parameter) of the first object [m].
    V0 : float
        Initial velocity magnitude of the first object along the
        x-axis [m/s].
    rf2 : array_like, shape (3,)
        Initial position vector of the second object in an inertial
        frame [m].
    vf2 : array_like, shape (3,)
        Initial velocity vector of the second object in an inertial
        frame [m/s].
    MU : float
        Standard gravitational parameter of the central body [m³/s²].

    Returns
    -------
    t_best : float
        Time since perigee at which the distance between the two
        objects is minimal [s].
    d_best : float
        Minimum Euclidean distance between the two objects over the
        search window [m].

    Notes
    -----
    - The closest-approach search is performed using a bisection-based
      minimization over a fixed time interval.
    - Motion is assumed to be purely Keplerian (two-body dynamics).
    - The first object's initial position is fixed at -200 AU on the
      x-axis.
    """

    # Initial state of object 1
    rf = np.array([-200 * AU, par[0], 0], dtype=float)
    vf = np.array([V0, 0, 0], dtype=float)


    # Ensure rf2 and vf2 are numpy float arrays
    rf2 = np.array(rf2, dtype=float)
    vf2 = np.array(vf2, dtype=float)

    rf, vf, info = state_at_perigee_from_rv(rf, vf, MU)

    t_best, d_best = minimize_distance_bisection(
        rf, vf, rf2, MU, T=8 * 86400  # search window
    )
    return t_best, d_best


def distance_vector(par, V0, rf2, MU):
    """
    Compute the minimum distance between two Keplerian trajectories
    over a fixed time window, using a bisection-based minimization.

    The first object is initialized on the Z=0 plane at the x-axis  
    initial -200 AU pos and a Y position given by par[0] an
    initial velocity along the +x direction. Its state is first
    propagated to perigee. The second object is defined by its
    initial Cartesian state (position and velocity).

    The function searches for the time of closest approach between
    the two objects within a given time window after perigee.
    
    Equivalent to distance_vector_gen but with one output to be used as
    a cost function in minimize process.

    Parameters
    ----------
    par : array_like
        Parameter vector. Only `par[0]` is used and represents the
        initial y-offset (impact parameter) of the first object [m].
    V0 : float
        Initial velocity magnitude of the first object along the
        x-axis [m/s].
    rf2 : array_like, shape (3,)
        Initial position vector of the second object in an inertial
        frame [m].
    MU : float
        Standard gravitational parameter of the central body [m³/s²].


    d_best : float
        Minimum Euclidean distance between the two objects over the
        search window [m].

    Notes
    -----
    - The closest-approach search is performed using a bisection-based
      minimization over a fixed time interval.
    - Motion is assumed to be purely Keplerian (two-body dynamics).
    - The first object's initial position is fixed at -200 AU on the
      x-axis.
    """
    
    # Initial state of object 1
    rf = np.array([-200 * AU, par[0], 0], dtype=float)
    vf = np.array([V0, 0, 0], dtype=float)

    # Ensure rf2 and vf2 are numpy float arrays
    rf2 = np.array(rf2, dtype=float)

    rf, vf, info = state_at_perigee_from_rv(rf, vf, MU)

    t_best, d_best = minimize_distance_bisection(
        rf, vf, rf2, MU, T=8 * 86400  # search window
    )
    return d_best


## Ensemble de fonction pour trouver le rapprochement maximal d'une orbite avec un point
def dist_at_time(t, rf0, vf0, rf2, MU):
    """
    Compute the distance between a propagated spacecraft position and a reference point at a given time.

    The function propagates an initial state vector (position and velocity) using
    Lagrangian propagation and returns the Euclidean distance between the propagated
    position and a fixed reference position.

    Parameters
    ----------
    t : float
        Propagation time since the initial epoch [s].
    rf0 : array_like, shape (3,)
        Initial position vector in an inertial frame [m].
    vf0 : array_like, shape (3,)
        Initial velocity vector in an inertial frame [m/s].
    rf2 : array_like, shape (3,)
        Reference position vector (target point) in the same frame [m].
    MU : float
        Standard gravitational parameter of the central body [m³/s²].

    Returns
    -------
    float
        Euclidean distance between the propagated position and the reference point [m].

    Notes
    -----
    - The propagation is performed using Lagrange coefficients via
      `pk.propagate_lagrangian`.
    """
    rf, vf = pk.propagate_lagrangian(rf0, vf0, t, MU)
    return np.linalg.norm(np.array(rf) - np.array(rf2))


def bracket_minimum(f, rf0, vf0, rf2, MU, T, N=200):
    """
    Bracket the minimum of a scalar function over a time interval.

    The function samples a scalar cost function (typically a distance)
    over a symmetric time window [-T, T], identifies the discrete minimum,
    and returns a triplet of times that brackets this minimum. 

    Parameters
    ----------
    f : callable
        Scalar function to be minimized. Must have the signature
        `f(t, rf0, vf0, rf2, MU)` and return a float.
    rf0 : array_like, shape (3,)
        Initial position vector of the propagated object [m].
    vf0 : array_like, shape (3,)
        Initial velocity vector of the propagated object [m/s].
    rf2 : array_like, shape (3,)
        Reference position vector (target point) [m].
    MU : float
        Standard gravitational parameter of the central body [m³/s²].
    T : float
        Half-width of the search interval in time [s].
        The function is evaluated over [-T, T].
    N : int, optional
        Number of sampling points in the interval (default is 200).

    Returns
    -------
    t_left : float
        Time just before the sampled minimum.
    t_mid : float
        Time at which the sampled minimum occurs.
    t_right : float
        Time just after the sampled minimum.

    Notes
    -----
    - This routine performs a coarse, discrete search and does not guarantee
      that the true continuous minimum lies inside the returned bracket if
      the sampling resolution is insufficient.
    - Increasing `N` improves robustness at the cost of additional function
      evaluations.
    """
    ts = np.linspace(-T, T, N)
    ds = []

    for t in ts:
        ds.append(f(t, rf0, vf0, rf2, MU))

    i = np.argmin(ds)

    # ensure we have neighbors
    if i == 0:
        i = 1
    if i == N - 1:
        i = N - 2

    return ts[i - 1], ts[i], ts[i + 1]  # bracket (t_left, t_mid, t_right)


def minimize_distance_bisection(rf0, vf0, rf2, MU, T, tol=1e-6):

    f = dist_at_time
    tL, tM, tR = bracket_minimum(f, rf0, vf0, rf2, MU, T)

    # Dichotomy on interval [tL, tR]
    while abs(tR - tL) > tol:
        t1 = tL + (tR - tL) / 3.0
        t2 = tR - (tR - tL) / 3.0

        d1 = f(t1, rf0, vf0, rf2, MU)
        d2 = f(t2, rf0, vf0, rf2, MU)

        if d1 < d2:
            tR = t2
        else:
            tL = t1

    t_best = 0.5 * (tL + tR)
    return t_best, f(t_best, rf0, vf0, rf2, MU)


def state_at_perigee_from_rv(r, v, mu, e_tol=1e-8, h_tol=1e-12):
    """
    Compute position and velocity at periapsis from Cartesian state (r, v) without
    calling high-level element conversions.
    Returns (r_peri, v_peri, info_dict).

    Behavior:
      - if eccentricity e >= e_tol: returns periapsis state (closest approach).
      - if e < e_tol (quasi-circular): returns (r, v) (no unique periapsis) and info['reason']='circular'.
      - raises ValueError if the input is degenerate (h ~ 0).
    Inputs:
      r, v : iterable of length 3 (numpy arrays accepted)
      mu   : gravitational parameter (same units as r^3 / t^2)
    """
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)
    r_norm = np.linalg.norm(r)
    v_norm = np.linalg.norm(v)

    # Angular momentum
    h_vec = np.cross(r, v)
    h_norm = np.linalg.norm(h_vec)
    if h_norm <= h_tol:
        raise ValueError(
            "Angular momentum near zero (radial motion) — cannot define orbital plane."
        )

    h_hat = h_vec / h_norm

    # Eccentricity vector
    e_vec = (np.cross(v, h_vec) / mu) - (r / r_norm)
    e = np.linalg.norm(e_vec)

    # quasi-circular case
    if e < e_tol:
        # no unique perigee: return current state and diagnostic
        return (
            r.copy(),
            v.copy(),
            {"e": e, "reason": "quasi-circular (no unique periapsis)"},
        )

    e_hat = e_vec / e

    # semi-latus rectum
    p = (h_norm**2) / mu

    # periapsis radius
    r_peri = p / (1.0 + e)

    # periapsis position (pointing along e_hat)
    r_p = r_peri * e_hat

    # transverse unit vector in orbital plane (so that e_hat, t_hat, h_hat form right-handed triad)
    t_hat = np.cross(h_hat, e_hat)
    t_hat_norm = np.linalg.norm(t_hat)
    if t_hat_norm == 0:
        # numerical degeneracy (shouldn't happen if h_norm>0 and e>0)
        raise RuntimeError("Failed to build transverse direction (degenerate).")
    t_hat = t_hat / t_hat_norm

    # velocity magnitude at periapsis: v_t = h / r_p (radial component 0)
    v_p = (h_norm / r_peri) * t_hat

    info = {"e": e, "p": p, "r_peri": r_peri, "h": h_norm, "e_vec": e_vec}
    return r_p, v_p, info


# Calcul fonction objectif
# ----------------------------
# Table des poids scientifiques (Table 1 du problème)
# ----------------------------
BODY_WEIGHTS = {
    1: 0.1,  # Vulcan
    2: 1,  # Yavin
    3: 2,  # Eden
    4: 3,  # Hoth
    1000: 5,  # Yandi
    5: 7,  # Beyoncé
    6: 10,  # Bespin
    7: 15,  # Jotunn
    8: 20,  # Wakonyingo
    9: 35,  # Rogue1
    10: 50,  # PlanetX
}
# Asteroids (1001–1257) → w = 1
for k in range(1001, 1258):
    BODY_WEIGHTS[k] = 1
# Comets (2001–2042) → w = 3
for k in range(2001, 2043):
    BODY_WEIGHTS[k] = 3


# ----------------------------
# 3.1 Grand tour bonus b
# ----------------------------
def grand_tour_bonus(flybys):
    """
    flybys: list of dicts with keys 'body_id' and 'is_science'
    """
    planets_and_dwarf = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1000]
    bodies_done = {f["body_id"] for f in flybys if f["is_science"]}
    asteroids_comets = [
        f["body_id"]
        for f in flybys
        if f["is_science"]
        and (1001 <= f["body_id"] <= 1257 or 2001 <= f["body_id"] <= 2042)
    ]
    if (
        all(pid in bodies_done for pid in planets_and_dwarf)
        and len(asteroids_comets) >= 13
    ):
        return 1.2
    return 1.0


# ----------------------------
# 3.4 Flyby velocity penalty F(Vinf)
# ----------------------------
def F_velocity(Vinf):
    """
    Vinf : vitesse hyperbolique en km/s
    """
    return 0.2 + np.exp(-Vinf / 13) / (1 + np.exp(-5 * (Vinf - 1.5)))


# ----------------------------
# 3.3 Seasonal penalty S(rhats)
# ----------------------------
def S_seasonal(rhats):
    """
    rhats : liste des vecteurs unitaires heliocentriques r̂_k,i (np.array(3,))
    """
    i = len(rhats)
    if i == 1:
        return 1.0
    s_sum = 0.0
    r_i = rhats[-1]
    for j in range(i - 1):
        r_j = rhats[j]
        cos_theta = np.dot(r_i, r_j)
        theta_deg = np.degrees(np.arccos(cos_theta))
        s_sum += np.exp(-(theta_deg**2) / 50.0)
    return 0.1 + 0.9 / (1 + 10 * s_sum)


# ----------------------------
# Objective function J (section 3)
# ----------------------------
def objective(flybys):
    """
    flybys: list of dicts with keys:
        - 'body_id': int
        - 'r_hat': np.array(3,)
        - 'Vinf': float (km/s)
        - 'is_science': bool
    """
    b = grand_tour_bonus(flybys)
    c = 1

    # take only science flybys
    flybys_by_body = defaultdict(list)
    for f in flybys:
        if f["is_science"]:
            flybys_by_body[f["body_id"]].append(f)

    J_total = 0.0
    for body_id, body_flys in flybys_by_body.items():
        w = BODY_WEIGHTS.get(body_id, 0)
        rhats = []
        for f in body_flys[:13]:  # max 13 flybys scientifiques par corps
            rhats.append(f["r_hat"])
            S = S_seasonal(rhats)
            F = F_velocity(f["Vinf"])
            J_total += w * S * F

    return J_total * b * c


def rotate_towards(v1, v2, alpha, eps=1e-12):
    """
    Rotate v1 by angle alpha towards v2 in 3D.
    Returns a vector with same magnitude as v1.
    """
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    if v1_norm < eps or v2_norm < eps:
        raise ValueError("v1 and v2 must be non-zero vectors")

    # normalize
    v1_hat = v1 / v1_norm
    v2_hat = v2 / v2_norm

    # rotation axis
    n = np.cross(v1_hat, v2_hat)
    n_norm = np.linalg.norm(n)

    if n_norm < eps:
        # Colinear or anti-colinear
        if np.dot(v1_hat, v2_hat) > 0:
            # same direction: no rotation
            return v1.copy()
        else:
            # opposite direction: pick any orthogonal axis
            arbitrary = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(v1_hat, arbitrary)) > 0.9:
                arbitrary = np.array([0.0, 1.0, 0.0])
            n = np.cross(v1_hat, arbitrary)
            n /= np.linalg.norm(n)
    else:
        n /= n_norm

    # Rodrigues rotation formula
    v_rot = (
        v1_hat * np.cos(alpha)
        + np.cross(n, v1_hat) * np.sin(alpha)
        + n * np.dot(n, v1_hat) * (1 - np.cos(alpha))
    )

    return v_rot * v1_norm


def norm(v):
    return np.linalg.norm(v)


def cos_angle_between(v1, v2):
    """Retourne le cosinus entre deux vecteurs 3D (en radians)."""
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    dot = np.dot(v1, v2)
    normi = norm(v1) * norm(v2)
    if normi == 0:
        raise ValueError("Un des vecteurs a une norme nulle")
    # Clamp pour éviter les erreurs numériques (acos argument hors [-1,1])
    return np.clip(dot / normi, -1.0, 1.0)


def calc_vmax_no_decel(v0, v_nom, a, T):
    """
    v0 : vitesse initiale réelle
    v_nom : vitesse nominale voulue
    a : accélération constante
    T : temps total
    """
    if a <= 0:
        raise ValueError("a must be positive")

    # coefficients quadratique : t_a^2 - 2*T*t_a + 2*T*(v_nom - v0)/a = 0
    A = 1.0
    B = -2 * T
    C = 2 * T * (v_nom - v0) / a

    discriminant = B**2 - 4 * A * C
    if discriminant < 0:
        return np.nan

    sqrtD = np.sqrt(discriminant)
    t_a1 = (-B + sqrtD) / (2 * A)
    t_a2 = (-B - sqrtD) / (2 * A)

    # prendre solution physique
    t_a = next((t for t in (t_a1, t_a2) if 0 <= t <= T), np.nan)
    if np.isnan(t_a):
        return np.nan

    v_max = v0 + a * t_a
    return v_max




# ---- FILTRE LAMBRET-VOILE ----


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

    Retourne True si la solution Lambert (v_lam0 à t0->tf) semble réalisable par la voile.

    - r1: function(t) -> position vector departure (m)
    - r2: function(t) -> position vector arrival (m)
    - t: transfer time (s)
    - vin_fb : vecteur vitesse initial avant le flyby précédédent l'arc de lambert courant (m/s)
    - v_init_lmbrt : vitesse demandé par Lambert au départ (m/s)
    - C : Altaira flux 𝐶 at 1 AU
    - r0 : Reference distance 𝑟0 km
    - A : sail area (m^2)
    - m : spacecraft mass
    - mu : star mu
    - planet : planet to flyby pykep object
    - vp : planet velocity heliocentric vector at flyby time

    - gamma : demi-angle du cone d'orientation admissible (degrés)

    - Nsamples : nb d'échantillons le long de l'arc
    - GA_samples : nb de test de dv fournit par le flyby précédent
    - Anomaly_samples : nb of evaluation point for efficiency computation
    - dv_margin_coef : marge en DV utilisée dans le filtrage



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
    # print(delta_ineq)
    if np.isnan(delta_ineq):
        delta_ineq = 0
        # print('delta ineq was nan')

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


def save_flybys_to_csv(flybys, filename="flybys.csv"):
    """
    Sauvegarde la liste de flybys dans un fichier CSV.
    Chaque élément de 'flybys' est un dictionnaire contenant des champs comme :
    'body_id', 'r_hat', 'Vinf', 'is_science', 'r2', 'v2', 'tof', etc.
    """
    if len(flybys) == 0:
        print("pas de flyby a sauvegarder")
        return

    # Collecter toutes les clés présentes dans au moins un flyby
    all_keys = set()
    for fb in flybys:
        all_keys.update(fb.keys())
    fieldnames = list(all_keys)

    # Écrire le fichier CSV
    with open(filename, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for fb in flybys:
            fb_serialized = {
                k: (
                    np.array2string(fb[k], separator=";", precision=15)
                    if isinstance(fb[k], np.ndarray)
                    else fb[k]
                )
                for k in fb
            }
            writer.writerow(fb_serialized)


def load_flybys_from_csv(filename="flybys.csv"):
    """
    Charge un fichier CSV de flybys avec parsing automatique :
    - Vecteurs : [1 2 3], [1;2;3], [1, 2, 3], (1.0, 2.0, 3.0)
    - Booléens : True / False
    - Nombres : int ou float
    - Sinon : string
    """
    flybys = []
    try:
        with open(filename, mode="r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fb = {}
                for key, val in row.items():
                    if val is None:
                        fb[key] = None
                        continue

                    val = val.strip()
                    if val == "":
                        fb[key] = None
                        continue

                    # --- BOOL ---
                    if val.lower() in ["true", "false"]:
                        fb[key] = val.lower() == "true"
                        continue

                    # --- VECTEUR (entre crochets ou parenthèses) ---
                    if (val.startswith("[") and val.endswith("]")) or (
                        val.startswith("(") and val.endswith(")")
                    ):
                        clean = val.strip("[]() ")
                        # remplace tous séparateurs possibles par des espaces
                        clean = re.sub(r"[;,]", " ", clean)
                        try:
                            arr = np.fromstring(clean, sep=" ")
                            if arr.size > 0:
                                fb[key] = arr
                                continue
                        except Exception:
                            pass

                    # --- FLOAT ---
                    try:
                        fb[key] = float(val)
                        continue
                    except ValueError:
                        pass

                    # --- INT ---
                    try:
                        fb[key] = int(val)
                        continue
                    except ValueError:
                        pass

                    # --- AUTRE (string brut) ---
                    fb[key] = val

                flybys.append(fb)

    except FileNotFoundError:
        print(f" Fichier '{filename}' introuvable.")
    return flybys


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


def write_gtoc13_solution(filename, output_table):
    """
    Écrit un tableau de sortie GTOC13 dans un fichier texte ASCII conforme.

    Parameters
    ----------
    filename : str
        Nom du fichier de sortie (.txt)
    output_table : list of tuples or lists
        Chaque élément doit être :
        (body_id, flag, epoch, r, v, u)
        où r, v, u sont des np.array ou listes de 3 floats
    """
    with open(filename, "w") as f:
        f.write("# GTOC13 trajectory solution file\n")
        f.write("# Columns: body_id, flag, epoch, rx, ry, rz, vx, vy, vz, ux, uy, uz\n")

        for entry in output_table:
            line, body_id, flag, epoch, r, v, u = entry

            # Vérification des longueurs
            r = np.asarray(r)
            v = np.asarray(v)
            # u = np.asarray(u) # no we dont want to make array if its not it kills the check after

            # Si u contient un unique 0 → remplacer par [0, 0, 0]
            if np.isscalar(u) and u == 0:
                u = np.zeros(3)
            elif len(u) == 1 and u[0] == 0:
                u = np.zeros(3)

            # Ligne complète (12 champs)
            line = (
                f"{int(body_id):d} {int(flag):d} {epoch:.17f} "
                f"{r[0]:.17f} {r[1]:.17f} {r[2]:.17f} "
                f"{v[0]:.17f} {v[1]:.17f} {v[2]:.17f} "
                f"{u[0]:.17f} {u[1]:.17f} {u[2]:.17f}\n"
            )

            f.write(line)

    print(
        f"Fichier '{filename}' écrit avec {len(output_table)} lignes conformes GTOC13."
    )


def min_radius_calc(r1, v1, r2, v2, tof):
    el_dep = list(pk.ic2par(r1, v1, MU_ALTAIRA))
    el_ar = list(pk.ic2par(r2, v2, MU_ALTAIRA))

    # Hyperbolic
    if el_dep[1] > 1:
        if np.sign(el_dep[5]) != np.sign(el_ar[5]):
            el_dep[5] = 0
        else:
            el_dep[5] = min([el_dep[5], el_ar[5]], key=abs)
        rmin, v = pk.par2ic(el_dep, MU_ALTAIRA)
        return np.linalg.norm(rmin)

    # Elliptic
    else:
        if el_dep[5] < 0:
            el_dep[5] = el_dep[5] + np.pi * 2
        if el_ar[5] < 0:
            el_ar[5] = el_ar[5] + np.pi * 2
        if el_ar[5] < el_dep[5] or tof > 2 * np.pi * np.sqrt(
            el_dep[0] ** 3 / MU_ALTAIRA
        ):
            el_dep[5] = 0
        else:
            E_closest = min([el_dep[5], el_ar[5]], key=dist_to_periapsis)
            el_dep[5] = E_closest

        rmin, v = pk.par2ic(el_dep, MU_ALTAIRA)
        return np.linalg.norm(rmin)


def dist_to_periapsis(nu):
    return min(abs(nu), abs(2 * np.pi - nu))


def build_sequence(
    J,
    t,
    flybys,
    bodies,
    min_tof,
    max_tof,
    tof_tries_nb,
    t_max,
    max_revs_lmbrt,
    bodies_to_parse,
    orbital_period_search = False,
    body_excluded=None,
    only_conics = False,
):
    
    # Initial state 
    r1 = flybys[-1]["r2"]
    vin_fb = flybys[-1]["v2"]
    
    ## Début de la recherche de la séquence
    print("begining optimal sequence search")
    max_revs_lmbrt = 100

    shield_burned = flybys[-1]["shield_burned"]
    count = 0 
    while True:
        count = count+1
        print("we are at run", count)
        print("T mission is :", t / 86400 / 365.25, " years")
        print("J estimated is :", J)

        tof = np.linspace(min_tof, np.min([max_tof, t_max - t]), tof_tries_nb)
        print(tof)
        score_over_time = 0
        tof_best = np.nan

        # Planet data at begining of leg
        body_j_before_id = flybys[-1]["body_id"]
        body_j_before = bodies[body_j_before_id]
        r_beg, vp_dep = body_j_before.eph(t / 86400)
        for i in range(len(tof)):
            print( "TOF scanning step is", i, "over", range(len(tof)))
            
            # We only parse requested objects --> can be bodies (all) or only planets
            for body_id, body_j in bodies_to_parse.items():
                if body_id != body_excluded and body_id != flybys[-1]["body_id"]:
                    

                    # get r2 at t + tof --> planet at end of leg
                    r2_body_j, v = body_j.eph((t + tof[i]) / 86400)
                    
                    # In all Lambert solutions considered we loo for the min DV (considering min Altaira approach constraint)
                    dv_min_found = np.inf 
                    for w in range(2):
                        if w == 0:
                            l = pk.lambert_problem(
                                r1=r1,
                                r2=r2_body_j,
                                tof=tof[i],
                                mu=MU_ALTAIRA,
                                cw=False,
                                max_revs=max_revs_lmbrt,
                            )
                        else:
                            l = pk.lambert_problem(
                                r1=r1,
                                r2=r2_body_j,
                                tof=tof[i],
                                mu=MU_ALTAIRA,
                                cw=True,
                                max_revs=max_revs_lmbrt,
                            )

                        for k in range(len(l.get_v1())):
                            v1 = l.get_v1()[k]
                            v2 = l.get_v2()[k]

                            # filter sol and compute score if valid

                            # verif de la contrainte de proximité ALTAIRA
                            pre_burned_shield = False
                            rmin = min_radius_calc(r1, v1, r2_body_j, v2, tof[i])
                            rmin = np.array(rmin)
                            invalid = False
                            if rmin < 0.01 * AU:
                                invalid = True
                            elif rmin > 0.01 * AU and rmin < 0.05 * AU:
                                if shield_burned == False:
                                    pre_burned_shield = True
                                else:
                                    invalid = True

                            # Solar sail filtering
                            if np.any(np.isnan(v1)):
                                invalid = True

                            if invalid == False:

                                # Minimisation du dv à viser
                                # 0) Conversion to planet frame
                                vin_fb_plf = np.array(vin_fb) - np.array(vp_dep)
                                v_init_lmbrt_plf = np.array(v1) - np.array(vp_dep)

                                # 1) delta-v requis
                                dv_req = pk.fb_vel(
                                    vin_fb_plf, v_init_lmbrt_plf, body_j_before
                                )


                                if dv_req < dv_min_found:
                                    dv_min_found = dv_req 
                                
                    # If one solution is satisfying we mininimize DV with parameter tof --> f(tof) = DV
                    dv_req = dv_min_found
                    if dv_req < 200:
                        #print('body excluded is', flybys[-1]["body_id"])
                        #print('index is',i)
                        #print('begin a minimize dv with tof', tof[i], 'and body id', body_id)
                        #print(dv_req)
                        #dv = minimize_lmbrt_dv(tof[i],t,
                        #        r1,
                        #        body_j,
                        #        vin_fb,
                        #        vp_dep,
                        #        100,
                        #        body_j_before)
                        #print("ehhh macarena :", dv)
                           
                        res_dv = minimize(
                            minimize_lmbrt_dv,
                            tof[i],
                            args=(
                                t,
                                r1,
                                body_j,
                                vin_fb,
                                vp_dep,
                                100,
                                body_j_before,
                            ),
                            method="Nelder-Mead",
                            
                            options={
                                "maxiter": 2000,
                            },
                        )
                        #print("DV minimized is", res_dv.fun)

                        if res_dv.fun < dv_min_found and res_dv.x > 0:
                            tof_af_minimize = res_dv.x[0]
                            v1, v2, r2_body_j, dv_req = update_lmbrt_params(
                                r1,
                                tof_af_minimize,
                                t,
                                body_j,
                                vin_fb,
                                100,
                                body_j_before,
                                0,
                                0,
                            )
                        else: 
                            tof_af_minimize = tof[i]
                    else:
                        tof_af_minimize = tof[i]
                         
                    if dv_req < 20 and  dv_req > 1e-6 and not only_conics :
                        [status, dv_left, vout] = sail_filter_fast(
                            r1,
                            r2_body_j,
                            tof_af_minimize,
                            vin_fb,
                            v1,
                            C,
                            r0,
                            A,
                            m,
                            MU_ALTAIRA,
                            body_j_before,
                            vp_dep,
                        )
                    elif dv_req <= 1e-6 :
                        status = True
                    else:
                        status = False

                    # écart de l'arc
                    if dv_req <  1e-6:
                        #print("We have a conic arc !")
                        dv_left = [0, 0, 0]
                        vout = v1
                        status = True
                        
                        # We measure the anomaly flied around the arc, if superior to half a period its not acceptable 
                        if orbital_period_search:
                            el_orb_search = pk.ic2par(r1,v1,MU_ALTAIRA)
                            
                            # Mandatory direct shot
                            if el_orb_search[1] > 1:
                                status = False
                                T = np.inf
                            else:
                                T = 2*np.pi * np.sqrt(el_orb_search[0]**3/MU_ALTAIRA)
                                #if tof_af_minimize/T > 0.5:
                                #    status = False
                                #else:
                                #    status = True

                    if status is True:

                        # objective computation list (body, tof, J)
                        new_flyby = {
                            "body_id": body_id,
                            "r_hat": r2_body_j / np.linalg.norm(r2_body_j),
                            "Vinf": np.linalg.norm(
                                np.array(v2) - np.array(v)
                            ),
                            "is_science": True,
                            "r2": r2_body_j,
                            "v2": v2,
                            "tof": tof_af_minimize,
                            "dv_left": dv_left,
                            "vout": vout,
                            "v1": v1,
                            "shield_burned": shield_burned,
                        }
                        # Ajout à la liste
                        flybys.append(new_flyby)

                        # Calcul du score
                        J = objective(flybys)
                        flybys.pop()
                        
                        if not orbital_period_search:
                            # Criteria is score / time of leg 
                            score_over_time_new = (
                            J / tof_af_minimize
                            )
                        else:
                            # Criteria is min period
                            score_over_time_new = (
                            1/T 
                            )
                        if score_over_time_new > score_over_time:
                            score_over_time = score_over_time_new
                            tof_best = tof_af_minimize
                            best_flyby = new_flyby
                            J_best = J
                            r2_body_j_best = r2_body_j
                            vinfb_best = v2

        # Pick max J/tof (simplified choice for now)
        t = t + tof_best
        # Validate flyby or last leg has been reached
        if not np.isnan(tof_best):
            r1 = r2_body_j_best
            vin_fb = vinfb_best
            
            # MAJ of shield state 
            if pre_burned_shield == True:
                shield_burned = True 
            
            # Ajout à la liste
            flybys.append(best_flyby)
            # Calcul du score
            J = objective(flybys)
        else:
            break
         
        # The orbital period aimed has been reached so we exit this sequence search 
        # as the direct objectivehere is to reduce orbital period, not to maximize score
        if orbital_period_search and T < max_tof:
            break
        
    print("Planet sequence builded !")
    # saving flybys
    save_flybys_to_csv(flybys)
    return flybys


# Function use in the sequence search to transform a small dv into a minimum (from scan to minimize)
def minimize_lmbrt_dv(tof, t, r1, body_j, vin_fb, vp, max_revs_lmbrt, planet):

    r2, v = body_j.eph(float((t + tof) / 86400))
    
    dv_req_best = np.inf
    for w in range(2):
        if w == 0:
            l = pk.lambert_problem(
                r1=r1, r2=r2, tof=float(tof), mu=MU_ALTAIRA, cw=True, max_revs=max_revs_lmbrt
            )
        else:
            l = pk.lambert_problem(
                r1=r1, r2=r2, tof=float(tof), mu=MU_ALTAIRA, cw=False, max_revs=max_revs_lmbrt
            )

        for k in range(len(l.get_v1())):
            v1 = l.get_v1()[k]
            # 0) Conversion to planet frame
            vin_fb_plf = np.array(vin_fb) - np.array(vp)
            v_init_lmbrt_plf = np.array(v1) - np.array(vp)
            # 1) delta-v requis
            dv_req = pk.fb_vel(vin_fb_plf, v_init_lmbrt_plf, planet)
            if dv_req < dv_req_best:
                dv_req_best = dv_req
    return dv_req_best


# Reproduce get_min_dv but with the full outputs (outside the minimizer)
def update_lmbrt_params(
    r1, tof, t, body_j, vin_fb, max_revs_lmbrt, planet_init, period, n_period
):
    r2, v = body_j.eph((t + tof + period * n_period) / 86400)

    # get r1 at L
    r_dep, vp_dep = planet_init.eph((t + period * n_period) / 86400)

    dv_req_best = np.inf
    for w in range(2):
        if w == 0:
            l = pk.lambert_problem(
                r1=r1, r2=r2, tof=tof, mu=MU_ALTAIRA, cw=True, max_revs=max_revs_lmbrt
            )
        else:
            l = pk.lambert_problem(
                r1=r1, r2=r2, tof=tof, mu=MU_ALTAIRA, cw=False, max_revs=max_revs_lmbrt
            )

        for k in range(len(l.get_v1())):
            v1 = l.get_v1()[k]
            v2 = l.get_v2()[k]
            v_init_lmbrt = v1
            # 0) Conversion to planet frame
            vin_fb_plf = np.array(vin_fb) - np.array(vp_dep)
            v_init_lmbrt_plf = np.array(v_init_lmbrt) - np.array(vp_dep)
            # 1) delta-v requis
            dv_req = pk.fb_vel(vin_fb_plf, v_init_lmbrt_plf, planet_init)
            if dv_req < dv_req_best:
                dv_req_best = dv_req
                v1_best = v1
                v2_best = v2

    return v1_best, v2_best, r2, dv_req_best


# get min dv function (used in the minimizer)
def get_min_dv(
    par,
    t_list,
    MU_ALTAIRA,
    max_revs_lmbrt,
    shield_burned,
    planet_init,
    body_j,
    n_period,
    period,
    V0,
    VFMIN,
    L,
    pos_list,
):
    tof = float(par[0])
    if tof <= 0:
        tof = 1

    vin_fb, t, pos = get_vin_fb_t(
        par, V0, VFMIN, L, t_list, pos_list, planet_init, MU_ALTAIRA, period
    )

    dv_min_req = np.inf

    # get r1 at L
    r1, vp_dep = planet_init.eph(t / 86400)
    # get r2 at t + tof --> planet at end of leg
    r2_body_j, v = body_j.eph((t + tof + period * n_period) / 86400)

    l = pk.lambert_problem(
        r1=r1, r2=r2_body_j, tof=tof, mu=MU_ALTAIRA, cw=True, max_revs=100
    )

    for k in range(len(l.get_v1())):
        v1 = l.get_v1()[k]
        v2 = l.get_v2()[k]

        ##    # verif de la contrainte de proximité ALTAIRA
        ##    rmin = min_radius_calc(r1,v1,r2_body_j,v2,tof)
        ##    rmin = np.array(rmin)
        ##    invalid = False
        ##    if rmin < 0.01 * AU:
        ##      invalid = True
        ##    elif rmin > 0.01*AU and rmin < 0.05*AU:
        ##      if shield_burned == False:
        ##        shield_burned = True
        ##      else:
        ##        invalid = True
        ##
        ##    # Solar sail filtering
        ##    if np.any(np.isnan(v1)):
        ##      invalid = True
        ##

        ##invalid = False
        ##if invalid == False:
        # Minimisation du dv à viser
        # 0) Conversion to planet frame

        vin_fb_plf = np.array(vin_fb) - np.array(vp_dep)
        v_init_lmbrt_plf = np.array(v1) - np.array(vp_dep)

        # 1) delta-v requis

        dv_req = pk.fb_vel(vin_fb_plf, v_init_lmbrt_plf, planet_init)
        # eq,ineq = pk.fb_con(vin_fb_plf, v_init_lmbrt_plf, planet_init)

        if dv_req < dv_min_req:
            dv_min_req = dv_req

    return dv_min_req


def minimize_not_blocked(
    distance_vector_and_leg_cost,
    par,
    period,
    n_period,
    planet_init,
    body_j,
    MU_ALTAIRA,
    shield_burned,
    V0,
    VFMIN,
    L,
    t,
    pos_list,
    best_vin,
    best_t,
    preopt=False,
    par0=None,
):
    res_f = False
    bounds = [
        (None, None),  # pos_x / AU  → libre
        (None, None),  # pos_y / AU  → libre
        (5000, 50000),  # V0 Bounded
        (1.745329, 2.268928),  # L
        (1.0 / 1e8, 86400 * 365.25 * 10 / 1e8),  # tof
    ]
    res_f = minimize(
        distance_vector_and_leg_cost,
        par,
        args=(
            period,
            n_period,
            planet_init,
            body_j,
            MU_ALTAIRA,
            shield_burned,
            V0,
            VFMIN,
            L,
            t,
            pos_list,
            best_vin,
            best_t,
            preopt,
            par0,
        ),
        method="Nelder-mead",
        bounds=bounds,
        options={"maxiter": 10000},
    )  # 'xatol': 1e-18, 'fatol': 1e-5})
    return res_f


# function used to solve the first free dv transfer from the possible arrival states
def leg_cost(
    planets,
    t,
    MU_ALTAIRA,
    max_revs_lmbrt,
    shield_burned,
    planet_init,
    body_aimed,
    V0,
    VFMIN,
    L,
    pos_list,
):
    
    # Planet lists is randomized 
    items = list(planets.items())
    random.shuffle(items)


    MU = MU_ALTAIRA
    tofbest = np.inf
    best_res = np.inf
    period = planet_init.compute_period(pk.epoch(0))
    tol = 0.1
    res_f_best = np.inf
    for n_period in range(200):
        for body_id, body_j in items:
            if body_id != body_aimed:
                best_res = np.inf

                # minimize pb parameters are [tof, V0 (s/c velocity at mission beginning), L (orbital posiiton of planet Vulcan at encounter]
                bounds = [
                    (1.0, 86400 * 365.25 * 10),
                    (5000, 50000),
                    (1.745329, 2.268928),
                ]
                
                # Start with a random guess
                par_random = [np.random.uniform(low, high) for (low, high) in bounds]
                par = par_random
                res = minimize(
                    get_min_dv,
                    par,
                    args=(
                        t,
                        MU_ALTAIRA,
                        max_revs_lmbrt,
                        shield_burned,
                        planet_init,
                        body_j,
                        n_period,
                        period,
                        V0,
                        VFMIN,
                        L,
                        pos_list,
                    ),
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={"maxiter": 2000},
                ) 
            
                if res.fun < tol: 
                    print("resultat optim 1")
                    print(res.fun)
                    tofbest = res.x[0]
                    V0_best = res.x[1]
                    L_best = res.x[2]
                    body_j_best = body_j
                    body_id_best = body_id

                    best_vin, best_t, best_pos = get_vin_fb_t(
                        res.x,
                        V0,
                        VFMIN,
                        L,
                        t,
                        pos_list,
                        planet_init,
                        MU_ALTAIRA,
                        period,
                    )
                    best_res = res.fun
                    print("accceptable solution found")
                    print("tryig to transform into a perfect conic arc")
                    

                    # 2nd minimize sequence
                    # Les parametres sont Y0 Z0 V0 L(planet_init_state) tof(flight time from planet 1 to planet 2)

                    par = [
                        best_pos[1] / AU,
                        best_pos[2] / AU,
                        V0_best,
                        L_best,
                        tofbest / 1e8,
                    ]
                    preopt = True
                    par0 = par
                    res_f = run_with_timeout(
                        minimize_not_blocked,
                        (
                            distance_vector_and_leg_cost,
                            par,
                            period,
                            n_period,
                            planet_init,
                            body_j,
                            MU_ALTAIRA,
                            shield_burned,
                            V0,
                            VFMIN,
                            L,
                            t,
                            pos_list,
                            best_vin,
                            best_t,
                            preopt,
                            par0,
                        ),
                        timeout=60,
                    )
                    if res_f != False and res_f != None:            
                        print("preoptim cost")
                        print(res_f.fun)
                        par[4] = res_f.x[4]
                        
                        res_f = run_with_timeout(
                            minimize_not_blocked,
                            (
                                distance_vector_and_leg_cost,
                                par,
                                period,
                                n_period,
                                planet_init,
                                body_j,
                                MU_ALTAIRA,
                                shield_burned,
                                V0,
                                VFMIN,
                                L,
                                t,
                                pos_list,
                                best_vin,
                                best_t,
                                
                                
                            ),
                            timeout=60,
                        )
                        print("end_optim cost")
                        
                        if res_f != False and res_f != None:  
                            print(res_f.fun)

                            if res_f.fun < res_f_best:
                                res_f_best = res_f.fun
                                print("new minimum")
                                print(res_f_best)
                                if res_f_best < 1:

                                    print("On converge !")
                                    print(res_f.x)
                                    print(body_id)
                                    print(n_period)
                                    # Point initial
                                    r_init = [
                                        -200 * AU,
                                        res_f.x[0] * AU,
                                        res_f.x[1] * AU,
                                    ]
                                    v_init = [res_f.x[2], 0, 0]

                                    # Etat de la planète à la rencontre
                                    r0, v0 = planet_init.eph(0)
                                    Elp = np.array(pk.ic2eq(r0, v0, MU))
                                    el0 = np.array(pk.ic2par(r0, v0, MU))
                                    Elp[5] = res_f.x[3]
                                    rp, vp = pk.eq2ic(eq=list(Elp), mu=MU)

                                    # temps au premier passage
                                    tp_first_pass = time_of_flight(r0, v0, rp, vp, MU)

                                    # écart obtenu
                                    t_best, d_best = minimize_distance_bisection(
                                        r_init,
                                        v_init,
                                        rp,
                                        MU_ALTAIRA,
                                        T=365.25 * 200 * 86400,  # search window
                                    )

                                    # Etat du sc à la rencontre
                                    rf, vf = pk.propagate_lagrangian(
                                        r_init, v_init, t_best, MU_ALTAIRA
                                    )
                                    print("écart mesuré avec la planète après optim")
                                    print(np.array(rf) - np.array(rp))

                                    # Temps réel à la rencontre
                                    while tp_first_pass < t_best:
                                        tp_first_pass = tp_first_pass + period

                                    # On ajoute le n periode supplémentaire du aux choix de l'optim d'attendre
                                    t = tp_first_pass + period * n_period

                                    # valeurs pour le premier élément du dico flybys
                                    r1_fb1 = rf
                                    v1_fb1 = vf
                                    tof_fb1 = t_best
                                    # valeurs pour le second elem du dico flyby

                                    body_id_fb2 = body_id_best
                                    r2_body_j, v = body_j_best.eph(
                                        (t + res_f.x[4] * 1e8) / 86400
                                    )

                                    v1, v2, r2, dv_req_best = update_lmbrt_params(
                                        r1_fb1,
                                        res_f.x[4] * 1e8,
                                        tp_first_pass,
                                        body_j_best,
                                        v1_fb1,
                                        100,
                                        planet_init,
                                        period,
                                        n_period,
                                    )

                                    # As v1 can integrate a small velocity error along the arc the position error at arrival might blow, its better to obtain exact v1
                                    print("dv_req after calc :", dv_req_best)
                                    print("compared to minimizer outputs :", res_f.fun)

                                    rtest, vtest = body_j_best.eph(
                                        (res_f.x[4] * 1e8 + t) / 86400
                                    )
                                    print("valeur testé :", rtest)
                                    print("at time, :", res_f.x[4] * 1e8 + t)
                                    print("r outputed:", r2)

                                    # save data
                                    data = {
                                        "r_init": r_init,
                                        "v_init": v_init,
                                        "tp_first_pass": tp_first_pass,
                                        "t": t,
                                        "t_best": t_best,
                                    }
                                    np.save(
                                        "save_state.npy", data
                                    )  # écrase le fichier à chaque fois

                                    return (
                                        t,
                                        r_init,
                                        v_init,
                                        r1_fb1,
                                        v1_fb1,
                                        tof_fb1,
                                        body_id_fb2,
                                        r2_body_j,
                                        v1,
                                        v2,
                                        r2,
                                        res_f.x[4] * 1e8,
                                    )

    return (
        t,
        r_init,
        v_init,
        r1_fb1,
        v1_fb1,
        tof_fb1,
        body_id_fb2,
        r2_body_j,
        v1,
        v2,
        r2,
        res_f.x[4] * 1e8,
    )


def distance_vector_and_leg_cost(
    par,
    period,
    n_period,
    planet_init,
    body_j,
    MU,
    shield_burned,
    V0,
    VFMIN,
    Llist,
    t_list,
    pos_list,
    best_vin,
    best_t,
    preopt,
    par0,
):
    
    if preopt == True:
        rf = np.array([-200 * AU, par0[0] * AU, par0[1] * AU], dtype=float)
        vf = np.array([par0[2], 0, 0], dtype=float)
        L = par0[3]
    else:
        # Initial state of object 1
        rf = np.array([-200 * AU, par[0] * AU, par[1] * AU], dtype=float)
        vf = np.array([par[2], 0, 0], dtype=float)

        L = par[3]

    tof = par[4] * 1e8
    r0, v0 = planet_init.eph(0)
    Elp = np.array(pk.ic2eq(r0, v0, MU))
    Elp[5] = L
    rf2, vf2 = pk.eq2ic(eq=list(Elp), mu=MU)

    # Ensure rf2 and vf2 are numpy float arrays
    rf2 = np.array(rf2, dtype=float)
    vf2 = np.array(vf2, dtype=float)

    t_best, d_best = minimize_distance_bisection(
        rf, vf, rf2, MU, T=365.25 * 200 * 86400  # search window
    )
    

    tp_first_pass = time_of_flight(r0, v0, rf2, vf2, MU)
    
    

    while tp_first_pass < t_best:
        tp_first_pass = tp_first_pass + period

    # On ajoute le n periode supplémentaire du aux choix de l'optim d'attendre
    t = tp_first_pass + period * n_period
    #t2 = best_t + period * n_period
    
    


    #print("t_values :", t,t2)

    rin, vin_fb = pk.propagate_lagrangian(rf, vf, t_best, MU)

    
    #print("delta_v_in:",vin_fb,best_vin)
    dv_req = get_min_dv_real(
        tof, t, MU, 100, shield_burned, vin_fb, planet_init, body_j
    )
    #dv_req2 = get_min_dv_real(
    #    tof, t2, MU, 100, shield_burned, best_vin, planet_init, body_j
    #)
    #dv_req3 = get_min_dv_real(
    #    tof, t2, MU, 100, shield_burned, vin_fb, planet_init, body_j
    #)

   # print("dv_req_delta",dv_req,dv_req2,dv_req3)

    #Time.sleep(1000)


    return d_best / 100 + dv_req * 10000  #


# same as get_min_dv but no interpolation on any table(used to finalize the solution)
def get_min_dv_real(
    tof, t, MU_ALTAIRA, max_revs_lmbrt, shield_burned, vin_fb, planet_init, body_j
):

    tof = float(tof)
    if tof <= 0:
        tof = 1

    dv_min_req = np.inf

    # get r1 at L
    r1, vp_dep = planet_init.eph(t / 86400)
    # get r2 at t + tof --> planet at end of leg
    r2_body_j, v = body_j.eph((t + tof) / 86400)

    for w in range(2):
        if w == 0:
            l = pk.lambert_problem(
                r1=r1, r2=r2_body_j, tof=tof, mu=MU_ALTAIRA, cw=True, max_revs=100
            )
        else:
            l = pk.lambert_problem(
                r1=r1, r2=r2_body_j, tof=tof, mu=MU_ALTAIRA, cw=False, max_revs=100
            )

        for k in range(len(l.get_v1())):
            v1 = l.get_v1()[k]
            v2 = l.get_v2()[k]

            ##    # verif de la contrainte de proximité ALTAIRA
            ##    rmin = min_radius_calc(r1,v1,r2_body_j,v2,tof)
            ##    rmin = np.array(rmin)
            ##    invalid = False
            ##    if rmin < 0.01 * AU:
            ##      invalid = True
            ##    elif rmin > 0.01*AU and rmin < 0.05*AU:
            ##      if shield_burned == False:
            ##        shield_burned = True
            ##      else:
            ##        invalid = True
            ##
            ##    # Solar sail filtering
            ##    if np.any(np.isnan(v1)):
            ##      invalid = True
            ##

            ##invalid = False
            ## if invalid == False:
            # Minimisation du dv à viser
            # 0) Conversion to planet frame

            vin_fb_plf = np.array(vin_fb) - np.array(vp_dep)
            v_init_lmbrt_plf = np.array(v1) - np.array(vp_dep)

            # 1) delta-v requis

            dv_req = pk.fb_vel(vin_fb_plf, v_init_lmbrt_plf, planet_init)
            # eq,ineq = pk.fb_con(vin_fb_plf, v_init_lmbrt_plf, planet_init)

            if dv_req < dv_min_req:
                dv_min_req = dv_req

    return dv_min_req


### utility function to interpolate inside the arrival states data
def get_vin_fb_t(par, V0, VFMIN, L, t_list, pos_list, planet_init, MU, period):
    # ---- Prepare extended output vector (VFMIN + t + pos) ----
    t_col = np.asarray(t_list).reshape(-1, 1)  # (N,1)
    pos_arr = np.asarray(pos_list).reshape(-1, 3)  # (N,3)
    Y = np.concatenate([VFMIN, t_col, pos_arr], axis=1)  # (N,7)

    # ---- Build unique sorted axes ----
    L_unique = np.unique(L)
    V0_unique = np.unique(V0)

    # ---- Map original L,V0 indices to sorted axes ----
    L_idx_sorted = np.argsort(L)
    V0_idx_sorted = np.argsort(V0)

    # ---- Reshape Y into grid (assume original Y matches order of L,V0) ----
    M = len(L_unique)
    K = len(V0_unique)
    Y_grid = Y.reshape(M, K, 7)

    # ---- If V0 is descending, flip axis 1 ----
    if V0[0] > V0[-1]:
        Y_grid = Y_grid[:, ::-1, :]

    # ---- If L is descending, flip axis 0 ----
    if L[0] > L[-1]:
        Y_grid = Y_grid[::-1, :, :]

    # ---- Create interpolator ----
    interp_fun = RegularGridInterpolator(
        (L_unique, V0_unique), Y_grid, bounds_error=False, fill_value=None
    )

    # ---- Evaluate ----
    vals = interp_fun((par[2], par[1]))  # shape (7,)

    vin_fb = vals[0:3]
    t = vals[3]
    pos = vals[4:7]

    # On récupre tof_init qui doit être convertit en tp_first pass
    L = par[2]

    r0, v0 = planet_init.eph(0)
    Elp = np.array(pk.ic2eq(r0, v0, MU))
    el0 = np.array(pk.ic2par(r0, v0, MU))
    Elp[5] = L
    rf2, vf2 = pk.eq2ic(eq=list(Elp), mu=MU)

    tp_first_pass = time_of_flight(r0, v0, rf2, vf2, MU)

    while tp_first_pass < t:
        tp_first_pass = tp_first_pass + period

    return vin_fb, tp_first_pass, pos



def time_of_flight(r0, v0, rf, vf, mu):
    # vecteur spécifique
    h_vec = np.cross(r0, v0)
    h = np.linalg.norm(h_vec)

    # semi-grand axe a
    r0_norm = np.linalg.norm(r0)
    v0_norm = np.linalg.norm(v0)
    energy = 0.5 * v0_norm**2 - mu / r0_norm
    a = -mu / (2 * energy)  # semi-grand axe

    # orbite circulaire ?
    e_vec = (np.cross(v0, h_vec) / mu) - (r0 / r0_norm)
    e = np.linalg.norm(e_vec)

    # angle vraie
    nu0 = np.arctan2(np.dot(np.cross(e_vec, r0), h_vec) / h, np.dot(r0, e_vec))
    rf_norm = np.linalg.norm(rf)
    nuf = np.arctan2(np.dot(np.cross(e_vec, rf), h_vec) / h, np.dot(rf, e_vec))

    # différence d'anomalie (0..2pi)
    dnu = (nuf - nu0) % (2 * np.pi)

    # pour orbite circulaire, temps de vol = fraction de période
    if np.isclose(e, 0.0):
        period = 2 * np.pi * np.sqrt(a**3 / mu)
        tof = dnu / (2 * np.pi) * period
    else:
        # orbite elliptique : utiliser l'équation de Kepler
        # E = 2*arctan(sqrt((1-e)/(1+e)) * tan(nu/2))
        E0 = 2 * np.arctan(np.sqrt((1 - e) / (1 + e)) * np.tan(nu0 / 2))
        Ef = 2 * np.arctan(np.sqrt((1 - e) / (1 + e)) * np.tan(nuf / 2))
        Mf = Ef - e * np.sin(Ef)
        M0 = E0 - e * np.sin(E0)
        n = np.sqrt(mu / a**3)
        tof = (Mf - M0) / n
        if tof < 0:
            tof += 2 * np.pi / n

    return tof




def run_with_timeout(func, args=(), timeout=5):
    result = {}

    def target():
        try:
            result["value"] = func(*args)
        except Exception as e:
            result["error"] = e

    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        return None  # timed out

    if "error" in result:
        raise result["error"]

    return result.get("value", None)


def backup_current_folder():
    # 1) Timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 2) Nom du sous-dossier
    backup_dir = f"backup_{timestamp}"

    # 3) Crée le dossier
    os.makedirs(backup_dir, exist_ok=True)

    # 4) Extensions à copier
    exts = (".csv", ".txt", ".npy")

    # 5) Parcours du dossier courant
    for file in os.listdir("."):
        if file.lower().endswith(exts) and os.path.isfile(file):
            shutil.copy(file, os.path.join(backup_dir, file))
            print(f"Copied: {file} → {backup_dir}")

    print(f"\nBackup complete in folder: {backup_dir}")




def angle_between(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    cos_theta = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    # éviter les erreurs numériques (ex: cos > 1 à cause de roundoff)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.arccos(cos_theta)  # en radians
    

def shield_state(shield_burned, r1, v1, r2, v2, tof):
    # verif de la contrainte de proximité ALTAIRA
    pre_burned_shield = False
    rmin = min_radius_calc(r1, v1, r2, v2, tof)
    rmin = np.array(rmin)
    invalid = False
    if rmin < 0.01 * AU:
        invalid = True
    elif rmin > 0.01 * AU and rmin < 0.05 * AU:
        if shield_burned == False:
            shield_burned = True
        else:
            invalid = True
    return invalid, shield_burned




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