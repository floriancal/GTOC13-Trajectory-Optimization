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
from poliastro.twobody.orbit import Orbit
from astropy import units as u
from poliastro.bodies import Body
from constants import MU_ALTAIRA, AU, t_max, C, A, m, r0
from utilities import *




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


# Objective function calculation
# ----------------------------
# Scientific weigths values
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
    Vinf : hyperbolic velocity in m/s but converted to km/s as requested by the problem score computation
    """
    Vinf = Vinf / 1000
    return 0.2 + np.exp(-Vinf / 13) / (1 + np.exp(-5 * (Vinf - 1.5)))


# ----------------------------
# 3.3 Seasonal penalty S(rhats)
# ----------------------------
def S_seasonal(rhats):
    """
    rhats : unitary heliocentric vectors r̂_k,i (np.array(3,))
    """
    i = len(rhats)
    if i == 1:
        return 1.0
    s_sum = 0.0
    r_i = rhats[-1]
    for j in range(i - 1):
        r_j = rhats[j]
        cos_theta = np.clip(np.dot(r_i, r_j), -1.0, 1.0)
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
            F = F_velocity(np.abs(f["Vinf"]))

            J_total += w * S * F

    return J_total * b * c


def rotate_towards(v1, v2, alpha, eps=1e-12):
    """
    Rotate a 3D vector v1 by a given angle towards another vector v2.

    The rotation is performed in the plane spanned by v1 and v2, using
    Rodrigues' rotation formula. The resulting vector has the same
    magnitude as v1 and is rotated by an angle `alpha` in the direction
    of v2.

    If v1 and v2 are colinear:
    - If they point in the same direction, v1 is returned unchanged.
    - If they point in opposite directions, an arbitrary orthogonal
      rotation axis is selected.

    Parameters
    ----------
    v1 : array_like, shape (3,)
        Initial 3D vector to be rotated.
    v2 : array_like, shape (3,)
        Target 3D vector defining the rotation direction.
    alpha : float
        Rotation angle in radians. Positive values rotate v1 towards v2.
    eps : float, optional
        Numerical tolerance used to detect zero-norm or colinear vectors.
        Default is 1e-12.

    Returns
    -------
    v_rot : ndarray, shape (3,)
        Rotated vector with the same magnitude as v1.

    Raises
    ------
    ValueError
        If either v1 or v2 has a norm smaller than `eps`.

    Notes
    -----
    - The rotation axis is defined as the normalized cross product
      of v1 and v2.
    - This function does not clamp `alpha`; it is the caller's
      responsibility to ensure the angle is meaningful in the
      given context.

    Examples
    --------
    >>> v1 = np.array([1.0, 0.0, 0.0])
    >>> v2 = np.array([0.0, 1.0, 0.0])
    >>> rotate_towards(v1, v2, np.pi / 2)
    array([0., 1., 0.])
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
    """
    Return the cosine of the angle between two 3D vectors.

    The result is the normalized dot product of the vectors and is
    clamped to the interval [-1, 1] to avoid numerical issues when
    used with inverse trigonometric functions.

    Parameters
    ----------
    v1 : array_like, shape (3,)
        First 3D vector.
    v2 : array_like, shape (3,)
        Second 3D vector.

    Returns
    -------
    cos_theta : float
        Cosine of the angle between v1 and v2.

    Raises
    ------
    ValueError
        If either input vector has zero norm.
    """
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
    Compute the maximum velocity reached without a deceleration phase.

    This function assumes a constant positive acceleration followed by
    no deceleration, over a total time interval `T`. The acceleration
    duration is determined such that the average velocity over `T`
    matches the desired nominal velocity `v_nom`.

    The problem is solved by finding the physically valid root of a
    quadratic equation for the acceleration time.

    Parameters
    ----------
    v0 : float
        Initial actual velocity.
    v_nom : float
        Desired nominal (average) velocity over the time interval `T`.
    a : float
        Constant acceleration magnitude (must be positive).
    T : float
        Total duration of the motion.

    Returns
    -------
    v_max : float
        Maximum velocity reached during the motion.
        Returns NaN if no physically valid solution exists.

    Raises
    ------
    ValueError
        If the acceleration `a` is not strictly positive.

    Notes
    -----
    - The acceleration time `t_a` is constrained to the interval [0, T].
    - If the quadratic discriminant is negative or no root lies in
      [0, T], the function returns NaN.
    - The maximum velocity is given by v_max = v0 + a * t_a.
    """
    if a <= 0:
        raise ValueError("a must be positive")

    # quadratic coef : t_a^2 - 2*T*t_a + 2*T*(v_nom - v0)/a = 0
    A = 1.0
    B = -2 * T
    C = 2 * T * (v_nom - v0) / a

    discriminant = B**2 - 4 * A * C
    if discriminant < 0:
        return np.nan

    sqrtD = np.sqrt(discriminant)
    t_a1 = (-B + sqrtD) / (2 * A)
    t_a2 = (-B - sqrtD) / (2 * A)

    # if solution do not exist return nan
    t_a = next((t for t in (t_a1, t_a2) if 0 <= t <= T), np.nan)
    if np.isnan(t_a):
        return np.nan

    v_max = v0 + a * t_a
    return v_max


def min_radius_calc(r1, v1, r2, v2, tof):
    """
    Compute the minimum radius along an orbital arc between two states.

    This function calculates the closest approach distance (periapsis) along
    the conic section connecting the departure and arrival states. It handles
    both elliptic and hyperbolic trajectories and accounts for angular
    adjustments to ensure a meaningful closest approach.

    Parameters
    ----------
    r1 : array_like
        Position vector at departure.
    v1 : array_like
        Velocity vector at departure.
    r2 : array_like
        Position vector at arrival.
    v2 : array_like
        Velocity vector at arrival.
    tof : float
        Time of flight between the two states (seconds).

    Returns
    -------
    rmin_norm : float
        Norm of the minimum radius along the orbital arc (meters).
    more_thn_once : bool 
        do we reach the min more than once ?
    """
    # init to False
    more_thn_once = False
    
    Altaira = Body(parent=None, k=MU_ALTAIRA * u.m**3 / u.s**2, name="Altaira")
  
    rx = r1 * u.m
    vx = v1 * u.m / u.s 
    orb = Orbit.from_vectors(Altaira, rx, vx)
    el_dep = [orb.a.value*1000, orb.ecc.value, orb.inc.value, orb.raan.value,  orb.argp.value, orb.nu.value]
    
    rx = r2 * u.m
    vx = v2 * u.m / u.s 
    orb = Orbit.from_vectors(Altaira, rx, vx)
    el_ar = [orb.a.value*1000, orb.ecc.value, orb.inc.value, orb.raan.value,  orb.argp.value, orb.nu.value]
 
    # Hyperbolic
    if el_dep[1] > 1:
        if np.sign(el_dep[5]) != np.sign(el_ar[5]):
            el_dep[5] = 0
        else:
            el_dep[5] = min([el_dep[5], el_ar[5]], key=abs)
        rmin, v = pk.par2ic(el_dep, MU_ALTAIRA)      
        return np.linalg.norm(rmin), more_thn_once


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
            more_thn_once = True       
        else:
            E_closest = min([el_dep[5], el_ar[5]], key=dist_to_periapsis)
            el_dep[5] = E_closest
        
        
            
        rmin, v = pk.par2ic(el_dep, MU_ALTAIRA)
        return np.linalg.norm(rmin), more_thn_once

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
    orbital_period_search=False,
    body_excluded=None,
    only_conics=False,
):
    """
    Build an optimal sequence of interplanetary flybys for a mission.

    This function performs a search over candidate bodies and times of flight
    to construct a flyby sequence. It evaluates Lambert transfers, minimizes
    delta-v required to match incoming flyby velocities, and applies
    mission constraints such as Altaira proximity limits, shield usage, and
    optionally solar sail feasibility. The sequence is iteratively
    constructed by selecting the flyby with the best score at each leg.

    Parameters
    ----------
    J : float
        Initial mission score.
    t : float
        Current mission epoch (seconds).
    flybys : list of dict
        List of already planned flybys.
    bodies : dict
        Dictionary of all potential target bodies, keyed by body ID.
    min_tof : float
        Minimum time of flight for scanning a leg (seconds).
    max_tof : float
        Maximum time of flight for scanning a leg (seconds).
    tof_tries_nb : int
        Number of TOF steps to scan between min_tof and max_tof.
    t_max : float
        Maximum allowed mission time (seconds).
    max_revs_lmbrt : int
        Maximum number of revolutions allowed in Lambert solutions.
    bodies_to_parse : dict
        Subset of bodies to consider for sequence building.
    orbital_period_search : bool, optional
        If True, the search aims to minimize orbital period rather than score.
    body_excluded : hashable, optional
        Body ID to exclude from consideration in the current leg.
    only_conics : bool, optional
        If True, ignore non-conic solutions (skip solar sail / delta-v adjustments).

    Returns
    -------
    flybys : list of dict
        Updated list of flybys including the newly constructed sequence. Each
        flyby dictionary contains fields such as:
        - "body_id" : target body ID
        - "r_hat" : unit vector of position at encounter
        - "Vinf" : hyperbolic excess velocity magnitude
        - "is_science" : whether the flyby is considered a science target
        - "r2", "v2" : spacecraft position and velocity at encounter
        - "tof" : time of flight for this leg
        - "dv_left", "vout", "v1" : additional flyby-related outputs
        - "shield_burned" : flag if Altaira shield is used

    Notes
    -----
    - The function uses a combination of TOF scanning, Lambert solutions,
      delta-v minimization, and optional solar sail filtering.
    - Shields are automatically tracked if the minimum Altaira approach
      constraint is violated.
    - For orbital period optimization, the sequence stops once the desired
      period is achieved.
    - The sequence is saved to CSV at the end via `save_flybys_to_csv`.
    """

    # Initial state
    r1 = flybys[-1]["r2"]
    vin_fb = flybys[-1]["v2"]

    ## Début de la recherche de la séquence
    print("begining optimal sequence search")
    max_revs_lmbrt = 100

    shield_burned = flybys[-1]["shield_burned"]
    count = 0
    while True:
        count = count + 1
        print("we are at run", count)
        print("T mission is :", t / 86400 / 365.25, " years")
        print("J estimated is :", J)

        tof = np.linspace(min_tof, np.min([max_tof, t_max - t]), tof_tries_nb)
        score_over_time = 0
        tof_best = np.nan

        # Planet data at begining of leg
        body_j_before_id = flybys[-1]["body_id"]
        body_j_before = bodies[body_j_before_id]
        r_beg, vp_dep = body_j_before.eph(t / 86400)
        print("TOF scanning number of steps are", range(len(tof)))
        for i in range(len(tof)):

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

                            # verif de la contrainte de proximité ALTAIRA
                            invalid, _ = shield_state(
                                shield_burned, r1, v1, r2_body_j, v2, tof[i]
                            )

                            # Bad return of lambert solver
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
                                shield_burned,
                            ),
                            method="Nelder-Mead",
                            options={
                                "maxiter": 2000,
                            },
                        )

                        if res_dv.fun < dv_min_found and res_dv.x > 0:
                            tof_af_minimize = res_dv.x[0]
                            v1, v2, r2_body_j, dv_req, pre_burned_shield = (
                                update_lmbrt_params(
                                    r1,
                                    tof_af_minimize,
                                    t,
                                    body_j,
                                    vin_fb,
                                    100,
                                    body_j_before,
                                    0,
                                    0,
                                    shield_burned,
                                )
                            )
                            # Update of planet velocity after tof update with DV minimization
                            _, v = body_j.eph((t + tof_af_minimize) / 86400)
                        else:
                            tof_af_minimize = tof[i]
                    else:
                        tof_af_minimize = tof[i]

                    # Solar sail filtering
                    if dv_req < 20 and dv_req > 1e-6 and not only_conics:
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
                    elif dv_req <= 1e-6:
                        status = True
                    else:
                        status = False

                    # écart de l'arc
                    if dv_req < 1e-6:
                        dv_left = [0, 0, 0]
                        vout = v1
                        status = True

                        # We measure the anomaly flied around the arc, if superior to half a period its not acceptable
                        if orbital_period_search:
                            el_orb_search = pk.ic2par(r1, v1, MU_ALTAIRA)

                            # Mandatory direct shot
                            if el_orb_search[1] > 1:
                                status = False
                                T = np.inf
                            else:
                                T = (
                                    2
                                    * np.pi
                                    * np.sqrt(el_orb_search[0] ** 3 / MU_ALTAIRA)
                                )
                                # if tof_af_minimize/T > 0.5:
                                #    status = False
                                # else:
                                #    status = True

                    if status is True:

                        # objective computation list (body, tof, J)
                        new_flyby = {
                            "body_id": body_id,
                            "r_hat": r2_body_j / np.linalg.norm(r2_body_j),
                            "Vinf": np.linalg.norm(np.array(v2) - np.array(v)),
                            "is_science": True,
                            "r2": r2_body_j,
                            "v2": v2,
                            "tof": tof_af_minimize,
                            "dv_left": dv_left,
                            "vout": vout,
                            "v1": v1,
                            "shield_burned": pre_burned_shield,
                        }
                        # Ajout à la liste
                        flybys.append(new_flyby)

                        # Calcul du score
                        J = objective(flybys)
                        flybys.pop()

                        if not orbital_period_search:
                            # Criteria is score / time of leg
                            score_over_time_new = J / tof_af_minimize
                        else:
                            # Criteria is min period
                            score_over_time_new = 1 / T
                        if score_over_time_new > score_over_time:
                            score_over_time = score_over_time_new
                            tof_best = tof_af_minimize
                            best_flyby = new_flyby
                            J_best = J
                            r2_body_j_best = r2_body_j
                            vinfb_best = v2

                            # Shield state update
                            shield_burned = pre_burned_shield

        # Pick max J/tof (simplified choice for now)
        t = t + tof_best
        # Validate flyby or last leg has been reached
        if not np.isnan(tof_best):
            r1 = r2_body_j_best
            vin_fb = vinfb_best

            # Add to list
            flybys.append(best_flyby)
            # Score computation
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


# Same as get_min_dv but no interpolation on any table(used to finalize the solution)
def get_min_dv_real(tof, t, max_revs_lmbrt, shield_burned, vin_fb, planet_init, body_j):
    """
    Compute the minimum required delta-v for a transfer without interpolation.

    This function solves Lambert problems between the departure planet and the
    target body for a given time of flight and evaluates the required delta-v
    to match the incoming flyby velocity. Both clockwise and counterclockwise
    solutions are considered, and the minimum delta-v is returned.

    This version does not rely on any interpolated tables and is intended to
    finalize a previously identified solution.

    Parameters
    ----------
    tof : float
        Time of flight of the transfer (seconds).
    t : float
        Departure epoch (seconds).
    max_revs_lmbrt : int
        Maximum number of revolutions for the Lambert problem.
        (Currently not used.)
    shield_burned : bool
        Flag indicating whether the thermal shield has been burned.
        (Currently not used.)
    vin_fb : array_like
        Incoming velocity vector at flyby, expressed in the inertial frame.
    planet_init : object
        Departure planet ephemeris object providing position and velocity
        via planet_init.eph(t).
    body_j : object
        Target body ephemeris object providing position and velocity
        via body_j.eph(t).

    Returns
    -------
    dv_min_req : float
        Minimum delta-v required to connect the flyby state to the Lambert
        transfer.
    """
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

            # Altaira proximity check
            invalid, _ = shield_state(shield_burned, r1, v1, r2_body_j, v2, tof)

            if not invalid:

                # 0) Conversion to planet frame
                vin_fb_plf = np.array(vin_fb) - np.array(vp_dep)
                v_init_lmbrt_plf = np.array(v1) - np.array(vp_dep)

                # 1) delta-v requis
                dv_req = pk.fb_vel(vin_fb_plf, v_init_lmbrt_plf, planet_init)

                if dv_req < dv_min_req:
                    dv_min_req = dv_req

    return dv_min_req


# Function use in the sequence search to transform a small dv into a minimum (from scan to minimize)
def minimize_lmbrt_dv(
    tof, t, r1, body_j, vin_fb, vp, max_revs_lmbrt, planet, shield_burned
):
    """
    Compute the minimum delta-v required for a Lambert transfer at fixed time of flight.

    This function is used during the sequence search to refine a small delta-v
    estimate obtained from a scan into a local minimum. For a given time of
    flight, both clockwise and counterclockwise Lambert solutions are evaluated,
    and the minimum flyby matching delta-v is returned.

    Parameters
    ----------
    tof : float
        Time of flight of the Lambert transfer (seconds).
    t : float
        Departure epoch (seconds).
    r1 : array_like
        Spacecraft position at the start of the transfer.
    body_j : object
        Target body ephemeris object.
    vin_fb : array_like
        Incoming velocity vector at flyby (in inertial frame).
    vp : array_like
        Velocity of the planet at departure epoch.
    max_revs_lmbrt : int
        Maximum number of revolutions allowed in the Lambert problem.
    planet : object
        Planet object used for flyby delta-v computation.

    Returns
    -------
    dv_req_best : float
        Minimum delta-v required to match the Lambert transfer at the flyby.
    """
    r2, v = body_j.eph(float((t + tof) / 86400))

    dv_req_best = np.inf
    for w in range(2):
        if w == 0:
            l = pk.lambert_problem(
                r1=r1,
                r2=r2,
                tof=float(tof),
                mu=MU_ALTAIRA,
                cw=True,
                max_revs=max_revs_lmbrt,
            )
        else:
            l = pk.lambert_problem(
                r1=r1,
                r2=r2,
                tof=float(tof),
                mu=MU_ALTAIRA,
                cw=False,
                max_revs=max_revs_lmbrt,
            )

        for k in range(len(l.get_v1())):
            v1 = l.get_v1()[k]
            v2 = l.get_v2()[k]

            # Altaira proximity check
            invalid, _ = shield_state(shield_burned, r1, v1, r2, v2, tof)

            if not invalid:
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
    r1,
    tof,
    t,
    body_j,
    vin_fb,
    max_revs_lmbrt,
    planet_init,
    period,
    n_period,
    shield_burned,
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

            # Altaira proximity check
            invalid, _ = shield_state(shield_burned, r1, v1, r2, v2, tof)

            if not invalid:
                # 0) Conversion to planet frame
                vin_fb_plf = np.array(vin_fb) - np.array(vp_dep)
                v_init_lmbrt_plf = np.array(v1) - np.array(vp_dep)
                # 1) delta-v requis
                dv_req = pk.fb_vel(vin_fb_plf, v_init_lmbrt_plf, planet_init)
                if dv_req < dv_req_best:
                    dv_req_best = dv_req
                    v1_best = v1
                    v2_best = v2

    # Update shield state
    invalid, shield_burned = shield_state(shield_burned, r1, v1_best, r2, v2_best, tof)
    if invalid:
        raise Exception("shit")
    return v1_best, v2_best, r2, dv_req_best, shield_burned


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
    """
    Compute the minimum required flyby delta-v for a Lambert transfer leg.

    This function evaluates all admissible Lambert solutions between the
    departure planet and a target body over a given time of flight, and
    returns the minimum delta-v required at the flyby to match the Lambert
    departure velocity.

    Parameters
    ----------
    par : array_like
        Optimization parameter vector. The first element is the time of
        flight of the Lambert leg (s).
    t_list : array_like
        List of candidate times associated with the optimization variables.
    MU_ALTAIRA : float
        Gravitational parameter of the central body.
    max_revs_lmbrt : int
        Maximum number of revolutions allowed for the Lambert solution.
    shield_burned : bool
        Current thermal shield state (not used directly in this function,
        but kept for interface consistency).
    planet_init : pykep.planet
        Departure planet object.
    body_j : pykep.planet
        Target body object at the end of the Lambert leg.
    n_period : int
        Number of full orbital periods added for phasing.
    period : float
        Orbital period used for phasing (s).
    V0 : array_like, shape (3,)
        Initial spacecraft heliocentric velocity.
    VFMIN : array_like
        Minimum final velocity constraint or target.
    L : int
        Index of the current leg in the trajectory sequence.
    pos_list : array_like
        List of planet indices or positions defining the trajectory.

    Returns
    -------
    dv_min_req : float
        Minimum required delta-v magnitude at the flyby to match a valid
        Lambert departure solution. Returns +∞ if no valid solution exists.

    Notes
    -----
    - Velocities are converted to the planet-centered frame before
      computing the flyby delta-v requirement.
    - Only the minimum delta-v over all Lambert branches is retained.
    """

    tof = float(par[0])
    if tof <= 0:
        tof = 1

    vin_fb, t, pos = get_vin_fb_t(
        par, V0, VFMIN, L, t_list, pos_list, planet_init, MU_ALTAIRA, period
    )

    dv_min_req = 1e10

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

        # Altaira proximity check
        invalid, _ = shield_state(shield_burned, r1, v1, r2_body_j, v2, tof)

        if not invalid:

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
    """
    Run a non-blocking local optimization of a trajectory leg cost function.

    This function wraps a Nelder-Mead optimization of the provided cost
    function with predefined bounds and arguments. It is intended to be used
    in conjunction with a timeout mechanism to avoid blocking the global
    search when convergence is slow or impossible.

    Parameters
    ----------
    distance_vector_and_leg_cost : callable
        Cost function evaluating the trajectory leg.
    par : array_like
        Initial guess for the optimization parameters.
    period : float
        Orbital period of the departure planet.
    n_period : int
        Number of additional waiting periods before departure.
    planet_init : object
        Departure planet ephemeris object.
    body_j : object
        Target body ephemeris object.
    MU_ALTAIRA : float
        Gravitational parameter used for propagation.
    shield_burned : bool
        Flag indicating whether the thermal shield has been burned.
    V0, VFMIN, L, t, pos_list : array_like
        Precomputed arrival-state data passed to the cost function.
    best_vin, best_t : array_like
        Best incoming velocity and time from a previous optimization stage.
    preopt : bool, optional
        If True, run the optimization in pre-optimization mode.
    par0 : array_like or None, optional
        Reference parameter vector used when ``preopt`` is True.

    Returns
    -------
    res_f : OptimizeResult
        Result of the Nelder-Mead optimization.
    """
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
    """
    Solve the first free delta-v transfer from a set of possible arrival states.

    This function explores possible flyby arrival states and target bodies to
    determine a feasible first trajectory leg. It performs a sequence of 2
    optimizations:
    1) a coarse search using interpolated arrival states,
    2) a refinement step enforcing a true conic arc,


    The search is performed over multiple waiting periods and candidate target
    bodies until a converged solution is found.

    Parameters
    ----------
    planets : dict
        Dictionary of candidate target bodies indexed by body identifier.
    t : float
        Initial reference epoch (seconds).
    MU_ALTAIRA : float
        Gravitational parameter used for propagation and Lambert solves.
    max_revs_lmbrt : int
        Maximum number of revolutions allowed in Lambert problems.
    shield_burned : bool
        Flag indicating whether the thermal shield has been burned.
    planet_init : object
        Departure planet ephemeris object.
    body_aimed : hashable
        Identifier of the body to be excluded from the search.
    V0 : array_like
        Grid values of initial spacecraft velocity.
    VFMIN : array_like
        Incoming flyby velocity vectors associated with the arrival states.
    L : array_like
        Grid of planetary true longitudes at encounter.
    pos_list : array_like
        Arrival position vectors associated with the grid points.

    Returns
    -------
    result : tuple
        Tuple containing the optimized parameters of the first trajectory leg,
        including epochs, spacecraft states at flyby, target body identifier,
        Lambert transfer velocities, and time of flight.
    """
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
                    print("First optim results are :", res.fun)
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
                    # Parameters are Y0 Z0 V0 L(planet_init_state) tof(flight time from planet 1 to planet 2)
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
                        print("preoptim cost is :", res_f.fun)
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

                        if res_f != False and res_f != None:

                            if res_f.fun < res_f_best:
                                res_f_best = res_f.fun
                                print("new minimum found :", res_f_best)

                                if res_f_best < 1:

                                    print("Convergence acheived !")

                                    # Initial point
                                    r_init = [
                                        -200 * AU,
                                        res_f.x[0] * AU,
                                        res_f.x[1] * AU,
                                    ]
                                    v_init = [res_f.x[2], 0, 0]

                                    # Planet state at encounter
                                    r0, v0 = planet_init.eph(0)
                                    Elp = np.array(pk.ic2eq(r0, v0, MU))
                                    el0 = np.array(pk.ic2par(r0, v0, MU))
                                    Elp[5] = res_f.x[3]
                                    rp, vp = pk.eq2ic(eq=list(Elp), mu=MU)

                                    # Time at first pass
                                    tp_first_pass = time_of_flight(r0, v0, rp, vp, MU)

                                    # Obtained delta
                                    t_best, d_best = minimize_distance_bisection(
                                        r_init,
                                        v_init,
                                        rp,
                                        MU_ALTAIRA,
                                        T=365.25 * 200 * 86400,  # search window
                                    )

                                    # S/C state at encounter
                                    rf, vf = pk.propagate_lagrangian(
                                        r_init, v_init, t_best, MU_ALTAIRA
                                    )
                                    print(
                                        "Delta with planet position is [m]:",
                                        np.array(rf) - np.array(rp),
                                    )

                                    # Time at encounter
                                    while tp_first_pass < t_best:
                                        tp_first_pass = tp_first_pass + period

                                    # We add the aditionnal initial planet periods duration chosen by the optimizer
                                    t = tp_first_pass + period * n_period

                                    # Values for first element of flyby dict
                                    r1_fb1 = rf
                                    v1_fb1 = vf
                                    tof_fb1 = t_best

                                    # Values for second elements of flyby dict
                                    body_id_fb2 = body_id_best
                                    r2_body_j, v = body_j_best.eph(
                                        (t + res_f.x[4] * 1e8) / 86400
                                    )

                                    v1, v2, r2, dv_req_best, _ = update_lmbrt_params(
                                        r1_fb1,
                                        res_f.x[4] * 1e8,
                                        tp_first_pass,
                                        body_j_best,
                                        v1_fb1,
                                        100,
                                        planet_init,
                                        period,
                                        n_period,
                                        shield_burned,
                                    )

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
                                    )  # Erase the file each time

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
    """
    Compute the combined distance and delta-v cost for a trajectory leg.

    This function evaluates a cost composed of:
    - the minimum distance between a propagated trajectory and a reference
      planetary orbit,
    - the minimum required delta-v to connect the resulting flyby state to
      the next leg using a Lambert transfer.

    It supports a pre-optimization mode in which part of the parameter vector
    is frozen.

    Parameters
    ----------
    par : array_like
        Optimization parameter vector.
    period : float
        Orbital period of the departure planet.
    n_period : int
        Additional number of full orbital periods to wait before departure.
    planet_init : object
        Departure planet ephemeris object.
    body_j : object
        Target body ephemeris object.
    MU : float
        Gravitational parameter of the central body.
    shield_burned : bool
        Flag indicating whether the thermal shield has been burned.
    V0, VFMIN, Llist, t_list, pos_list : array_like
        Precomputed arrival-state data (not directly used in this function,
        kept for interface consistency).
    best_vin, best_t : array_like
        Best incoming velocity and time from a previous optimization stage
        (not directly used).
    preopt : bool
        If True, use fixed parameters from ``par0`` instead of ``par``.
    par0 : array_like
        Reference parameter vector used during pre-optimization.

    Returns
    -------
    cost : float
        Scalar cost combining normalized distance and delta-v penalties.
    """
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

    rin, vin_fb = pk.propagate_lagrangian(rf, vf, t_best, MU)

    dv_req = get_min_dv_real(tof, t, 100, shield_burned, vin_fb, planet_init, body_j)

    return d_best / 100 + dv_req * 10000


# Utility function to interpolate inside the arrival states data
def get_vin_fb_t(par, V0, VFMIN, L, t_list, pos_list, planet_init, MU, period):
    """
    Interpolate arrival-state data and compute the corresponding flyby timing.

    This function interpolates precomputed arrival data on a (L, V0) grid to
    obtain the incoming velocity at flyby, the target arrival epoch, and the
    spacecraft position. The time of flight from the initial planetary state
    is then adjusted to match the interpolated arrival time.

    Parameters
    ----------
    par : array_like
        Parameter vector. par[1] is the interpolation value for V0 and
        par[2] is the interpolation value for L.
    V0 : array_like
        Grid values of initial velocity parameter.
    VFMIN : array_like
        Incoming velocity vectors at flyby for each grid point, shape (N, 3).
    L : array_like
        Grid values of true longitude.
    t_list : array_like
        Arrival times associated with each grid point.
    pos_list : array_like
        Arrival position vectors, shape (N, 3).
    planet_init : object
        Planet ephemeris object providing initial position and velocity
        via planet_init.eph(t).
    MU : float
        Gravitational parameter of the central body.
    period : float
        Orbital period of the planet, used to wrap the time of flight.

    Returns
    -------
    vin_fb : ndarray
        Interpolated incoming velocity vector at flyby.
    tp_first_pass : float
        Time of flight from the initial state to the flyby, adjusted by integer
        multiples of the orbital period.
    pos : ndarray
        Interpolated spacecraft position at arrival.
    """
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
    """
    Compute the time of flight between two position vectors on the same Keplerian orbit.

    The time of flight is computed from the initial and final true anomalies.
    Circular and elliptical orbits are handled. Hyperbolic or parabolic
    trajectories are not supported.

    Parameters
    ----------
    r0 : array_like
        Initial position vector.
    v0 : array_like
        Initial velocity vector.
    rf : array_like
        Final position vector.
    vf : array_like
        Final velocity vector (not used, kept for interface consistency).
    mu : float
        Gravitational parameter of the central body.

    Returns
    -------
    tof : float
        Time of flight between r0 and rf along the orbit.
    """

    # h vector
    h_vec = np.cross(r0, v0)
    h = np.linalg.norm(h_vec)

    # semi-major axis
    r0_norm = np.linalg.norm(r0)
    v0_norm = np.linalg.norm(v0)
    energy = 0.5 * v0_norm**2 - mu / r0_norm
    a = -mu / (2 * energy)  # semi-grand axe

    # Circular orbit
    e_vec = (np.cross(v0, h_vec) / mu) - (r0 / r0_norm)
    e = np.linalg.norm(e_vec)

    # Anomaly
    nu0 = np.arctan2(np.dot(np.cross(e_vec, r0), h_vec) / h, np.dot(r0, e_vec))
    rf_norm = np.linalg.norm(rf)
    nuf = np.arctan2(np.dot(np.cross(e_vec, rf), h_vec) / h, np.dot(rf, e_vec))

    # Anomaly deltas (0..2pi)
    dnu = (nuf - nu0) % (2 * np.pi)

    # Simple calculation for circular orbits
    if np.isclose(e, 0.0):
        period = 2 * np.pi * np.sqrt(a**3 / mu)
        tof = dnu / (2 * np.pi) * period
    else:
        # Elliptic orbits : use Kepler equation
        E0 = 2 * np.arctan(np.sqrt((1 - e) / (1 + e)) * np.tan(nu0 / 2))
        Ef = 2 * np.arctan(np.sqrt((1 - e) / (1 + e)) * np.tan(nuf / 2))
        Mf = Ef - e * np.sin(Ef)
        M0 = E0 - e * np.sin(E0)
        n = np.sqrt(mu / a**3)
        tof = (Mf - M0) / n
        if tof < 0:
            tof += 2 * np.pi / n

    return tof


def shield_state(shield_burned, r1, v1, r2, v2, tof):
    """
    Evaluate thermal shield usage and proximity constraint violation.

    This function checks whether a trajectory segment violates the
    minimum allowed distance to the central star (ALTAIRA) and updates
    the thermal shield state accordingly.

    If the trajectory passes too close to the star, the solution is
    either invalidated or consumes the thermal shield, depending on
    the minimum distance reached and the current shield state.

    Parameters
    ----------
    shield_burned : bool
        Indicates whether the thermal shield has already been used.
    r1 : array_like, shape (3,)
        Initial position vector of the trajectory segment.
    v1 : array_like, shape (3,)
        Initial velocity vector of the trajectory segment.
    r2 : array_like, shape (3,)
        Final position vector of the trajectory segment.
    v2 : array_like, shape (3,)
        Final velocity vector of the trajectory segment.
    tof : float
        Time of flight of the trajectory segment (s).

    Returns
    -------
    invalid : bool
        True if the trajectory violates the minimum proximity constraint
        or requires a shield that has already been burned.
    shield_burned : bool
        Updated thermal shield state after evaluating this segment.

    Notes
    -----
    - If the minimum distance to the star is below 0.01 AU, the trajectory
      is immediately considered invalid.
    - If the minimum distance lies between 0.01 AU and 0.05 AU, the thermal
      shield is consumed if not already used; otherwise, the trajectory
      is invalid.
    - The minimum distance is computed using `min_radius_calc`.
    """
    rmin, more_thn_once = min_radius_calc(r1, v1, r2, v2, tof)
    rmin = np.array(rmin)
    invalid = False
    if rmin < 0.01 * AU:
        invalid = True
    elif rmin > 0.01 * AU and rmin < 0.05 * AU:
        if shield_burned == False and not more_thn_once:
            shield_burned = True
        else:
            invalid = True
    return invalid, shield_burned
