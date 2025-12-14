# Import needed modules
import numpy as np
from collections import defaultdict
import csv
import pykep as pk
import matplotlib.pyplot as plt
import orbital
from utilities import *
from scipy.optimize import minimize
import os
import pickle
import json
import random

print("Welcome to GTOC13 optimization program")


# Here we are in the main loop

# Hyperparams
tof_tries_nb = (
    2000  # Size of the times_of_flight vector to parse in the sequence finding
)
min_tof = 20000 * 86400  # Minimum tof between two planets [s]
max_tof = 100000 * 86400  # Maximum tof between two planets [s]
max_revs_lmbrt = 10  # In the Lambert solving algorithm there is 2N+1 solutions with N being the number of revolutions

V0_min = 3000  # This is the initial Min and Max Velocities in [m/s]
V0_max = 50000

# What to execute
initial_point_search = False  # Not to be runned each time, this builds a table of velocity/position and times vectors at first body encounter as a function of the initial state of the problem
first_two_legs_search = False # Wether to compute or reuse the choice of the two firsts conic legs
sequence_search = True  # Wether to run the flybys sequence search or not
solve_solar_sail_arcs = True  # Wether to run the solving of the sail arcs
only_refine = False  # Wether to only run the refinement of the solar sail solving of controlled arcs (the solving process is divided in two steps)
Write_GTOC_output_file = True  # Wether to prepare the GTOC format solution file or not

# Will re-run sequence search if no convergence and will restart all process forever with randomness to try improving the solution
Perpetual_run = True


# Initial Parameters
# The initial planet is always Vulcan to use at best the braking it can provide
body_aimed = 1
lambda0 = (
    np.pi / 2
)  # initial orbital position of Vulcan at encounter (can be tuned for varying results)
if Perpetual_run:
    sequence_search_depth = 3  # if a segment leads to no solution before the end of the mission, defines how many times we can remove the last flyby and replace with another
else:
    sequence_search_depth = 2  # smaller value for faster test/analysis


# Physical constants of the problem
MU_ALTAIRA = 139348062043.343 * 1e9
AU = 149597870691
t_max = 200 * 365.25 * 86400  # years expresses in seconds
C = 5.4026 * 10**-6  # N/m^2
A = 15.0  # m^2
m = 500.0  # kg
r0 = 149597870691  # m

# Clean tmp_file
if sequence_search == True:
    if os.path.exists("res_one_shot.pkl"):
        os.remove("res_one_shot.pkl")


# half of tolerance on position reqested in the pb (meters)
tol_pos = 50

# Loading all the objects
planets = load_bodies_from_csv("gtoc13_planets.csv")
asteroids = load_bodies_from_csv("gtoc13_asteroids.csv", mu_default=0.0)
comets = load_bodies_from_csv("gtoc13_comets.csv", mu_default=0.0)

# Fusion en un seul dictionnaire
bodies = {**planets, **asteroids, **comets}

# Initialisation du dictionnaire de flybys (used for score computations)
print("Problem initialized, bodies loaded.")


# Infinite run.
count = 0
while True:
    flybys = []
    
    # Here we run the initial mapping explained above
    if initial_point_search is True:
        filename = "init_pos.csv"
        # -------------------------------
        # RESET → delete existing file
        # -------------------------------
        if os.path.isfile(filename):
            os.remove(filename)  # <-- remise à zéro
        # ---------------------------------------------------------
        # If file does not exist → create it and write header
        # ---------------------------------------------------------
        file_exists = os.path.isfile(filename)

        if not file_exists:
            with open(filename, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "iter",
                        "t0_opt",
                        "tp_first_pass_opt",
                        "pos_0_opt_x",
                        "pos_0_opt_y",
                        "pos_0_opt_z",
                        "v_0_opt_x",
                        "v_0_opt_y",
                        "v_0_opt_z",
                        "rf_min_opt_x",
                        "rf_min_opt_y",
                        "rf_min_opt_z",
                        "vf_min_opt_x",
                        "vf_min_opt_y",
                        "vf_min_opt_z",
                        "body_aimed",
                        "cost",
                        "lambda0",
                    ]
                )

        print("Looking for an initial point..")
        while True:

            # Get body position at lambda
            planet_init = planets[body_aimed]
            rp, vp = planet_init.eph(0)
            el_p = pk.ic2par(rp, vp, MU_ALTAIRA)
            Ep = np.array(pk.ic2eq(r=rp, v=vp, mu=MU_ALTAIRA))
            Ep[5] = lambda0
            rp, vp = pk.eq2ic(eq=list(Ep), mu=MU_ALTAIRA)

            # Range of V0 to parse
            V_space = np.linspace(V0_max, V0_min, 100)

            par_start = [rp[1], 0]
            score_over_time = 0
            invalid = False
            for i in range(len(V_space)):
                print("Running start number :", i)

                res = minimize(
                    distance_vector,
                    par_start,
                    args=(V_space[i], rp, vp, MU_ALTAIRA),
                    method="Nelder-Mead",
                    options={
                        "disp": False,
                        "maxiter": 1000,
                        "xatol": 1e-18,
                        "fatol": 0.1,
                    },
                )

                par_start = res.x

                # We only keep converged points
                print("Result start number:", i)
                print("Optimal initial state pos:", res.x)
                print("Final cost:", res.fun)

                # is solution acceptable
                if res.fun < tol_pos:

                    # Xcheck with Keplerian propagation
                    rf = [-200 * AU, res.x[0], 0]
                    vf = [V_space[i], 0, 0]
                    rfper, vfper, info = state_at_perigee_from_rv(
                        rf, vf, MU_ALTAIRA
                    )

                    # Evaluate the propgation time
                    dt, d_best = distance_vector_gen(
                        res.x, V_space[i], rp, vp, MU_ALTAIRA
                    )
                    rf_min, vf_min = pk.propagate_lagrangian(
                        r0=rfper, v0=vfper, tof=dt, mu=MU_ALTAIRA
                    )
                    t0, tp_first_pass, tof_init = time_find(
                        res.x,
                        V_space[i],
                        rf_min,
                        vf_min,
                        rp,
                        vp,
                        el_p,
                        planet_init,
                    )

                    rf_min, vf_min = pk.propagate_lagrangian(
                        r0=rf, v0=vf, tof=tof_init, mu=MU_ALTAIRA
                    )

                    # Check ALTAIRA proximity constraint
                    rmin = min_radius_calc(rf, vf, rf_min, vf_min, tof_init)
                    rmin = np.array(rmin)

                    if rmin < 0.01 * AU:
                        invalid = True

                    rp0, vp0 = planet_init.eph(0)
                    rp, vp = pk.propagate_lagrangian(
                        r0=rp0, v0=vp0, tof=tp_first_pass, mu=MU_ALTAIRA
                    )
                    rdiff = np.linalg.norm(np.array(rp) - np.array(rf_min))
                    if invalid == False:
                        if rdiff > 50:
                            print(rp)
                            print(rf)
                            print(rf_min)
                            print(rdiff)
                            raise Exception(
                                "crosscheck of initial point conv not validated"  # If raise an error it shows a pb in the script
                            )

                    # store all data needed and write
                    pos_0_opt = [-200 * AU, res.x[0], 0]
                    v_0_opt = [V_space[i], 0, 0]
                    t0_opt = t0
                    tp_first_pass_opt = tp_first_pass
                    rf_min_opt = rf_min
                    vf_min_opt = vf_min
                    cost = res.fun
                    # Append one line per iteration
                    with open(filename, mode="a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(
                            [
                                i,
                                f"{t0_opt:.17e}",
                                f"{tp_first_pass_opt:.17e}",
                                f"{pos_0_opt[0]:.17e}",
                                f"{pos_0_opt[1]:.17e}",
                                f"{pos_0_opt[2]:.17e}",
                                f"{v_0_opt[0]:.17e}",
                                f"{v_0_opt[1]:.17e}",
                                f"{v_0_opt[2]:.17e}",
                                f"{rf_min_opt[0]:.17e}",
                                f"{rf_min_opt[1]:.17e}",
                                f"{rf_min_opt[2]:.17e}",
                                f"{vf_min_opt[0]:.17e}",
                                f"{vf_min_opt[1]:.17e}",
                                f"{vf_min_opt[2]:.17e}",
                                f"{body_aimed:.17e}",
                                f"{cost:.17e}",
                                f"{lambda0:17e}",
                            ]
                        )
            # Perturbate lambda0 by a small amount 
            lambda0 = (lambda0 + np.pi / 180 * 10) % (2 * np.pi)
        print("saving results to file 'init_pos.csv'")

    # Now that we either runned initial point table build or not we start here by defining the two first flybys             
    if sequence_search == True:
        
        if first_two_legs_search == True:
            # Reading the previously builded file
            shield_burned, V0_list, VFMIN_list, L_list, tp_list, pos_list = read_init_pos(
                "init_pos.csv"
            )

            # Convert to np arrays
            V0 = np.array(V0_list)  # shape = (N, 3)
            VFMIN = np.array(VFMIN_list)  # shape = (N, 3)
            L = np.array(L_list)
            tp = np.array(tp_list)

            # Solve the two first legs problem
            planet_init = planets[body_aimed]

            t, r_init, v_init, r1_fb1, v1_fb1, tof_fb1, body_id_fb2, r2_body_j, v1, v2, r2, tof2 = leg_cost(
                planets,
                tp,
                MU_ALTAIRA,
                max_revs_lmbrt,
                shield_burned,
                planet_init,
                body_aimed,
                V0,
                VFMIN,
                L,
                pos_list,
            )
            
            # two first legs are now obtained, prepare to compute the rest of the sequence 
            shield_burned = False
            # Proimity constraint calculation
            tof_for_min_radius = time_of_flight(r_init, v_init, r1_fb1, v1_fb1, MU_ALTAIRA)
            rmin = min_radius_calc(r_init, v_init, r1_fb1, v1_fb1, tof_for_min_radius)
            rmin = np.array(rmin)
            if rmin < 0.01 * AU:
                print(rmin)
                raise Exception(
                    "Rmin too small on first leg"
                )  # This is not supposed to happen and should be protected in leg_cost
            elif rmin > 0.01 * AU and rmin < 0.05 * AU:
                shield_burned = True

            if t < t_max:
                new_flyby = {
                    "body_id": body_aimed,
                    "r_hat": r1_fb1 / np.linalg.norm(r1_fb1),
                    "Vinf": np.linalg.norm(v1_fb1),
                    "is_science": True,
                    "r2": r1_fb1,
                    "v2": v1_fb1,
                    "tof": tof_fb1,
                    "dv_left": [0, 0, 0],
                    "vout": [0, 0, 0],
                    "v1": [0, 0, 0],
                    "shield_burned": shield_burned,
                }

                # Add to the flybies list
                flybys.append(new_flyby)
                # Score computation
                J = objective(flybys)

                if t < t_max:
                    # Second flyby
                    new_flyby = {
                        "body_id": body_id_fb2,
                        "r_hat": r2_body_j / np.linalg.norm(r2_body_j),
                        "Vinf": np.linalg.norm(v2),
                        "is_science": True,
                        "r2": r2,
                        "v2": v2,
                        "tof": tof2,
                        "dv_left": [0, 0, 0],
                        "vout": v1,
                        "v1": v1,
                        "shield_burned": shield_burned,
                    }
                    t = t + tof2
                    # Add to flyby list
                    flybys.append(new_flyby)
                    # Compute score
                    J = objective(flybys)
                    # saving flybys
                    save_flybys_to_csv(flybys, filename="flyby_first_two_legs.csv")

                else:
                    print("First segment is already > t_max : ", t / 86400 / 365.25)
                
                # Save t for next runs in addition to the flybys dict
                with open("t.pkl", "wb") as f:
                    pickle.dump(t, f)   
                
        else:
            flybys = load_flybys_from_csv(filename="flyby_first_two_legs.csv")
            J = objective(flybys)


                
        ### -------------- SEQUENCE SEARCH ----------------###      
        # Initial state is the state of the last flyby 
        with open("t.pkl", "rb") as f:
            t = pickle.load(f)
        r1 = flybys[-1]["r2"]
        v1_fb1 = flybys[-1]["v2"]
                
        flybys = build_sequence(
            J,
            t,
            flybys,
            bodies,
            min_tof,
            max_tof,
            tof_tries_nb,
            t_max,
            max_revs_lmbrt,
            r1,
            v1_fb1,
        )
        save_flybys_to_csv(flybys, filename="flybys_first_row.csv")


    # Or we reload past solution
    else:
        flybys = load_flybys_from_csv(filename="flybys.csv")
        J = 0 # Undefined for now but need to be initialized

    if solve_solar_sail_arcs:
        # Début du controle de la sail sur les segments
        print("Begining sail control arcs solving")
        save_path = "results_flybys.txt"
        save_path_json = "results_flybys.json"

        # --- Suppression du fichier en début de run ---
        if os.path.exists(save_path):
            os.remove(save_path)
            print(f"🗑️  Fichier {save_path} supprimé en début de run.")
        if os.path.exists(save_path_json):
            os.remove(save_path_json)
            print(f"🗑️  Fichier {save_path} supprimé en début de run.")

        for depth_nb in range(sequence_search_depth):
            # Re-run sequence search excluding last sol
            if depth_nb > 0:
                # on slice la liste
                flybys = flybys[: n_fb + 1]
                rin = flybys[-1]["r2"]
                vin = flybys[-1]["v2"]
                body_id_exclude = flyby["body_id"]

                data = np.load("save_state.npy", allow_pickle=True).item()
                tp_first_pass_opt = data["t"]
                t_best = data["t_best"]
                t0_opt = tp_first_pass_opt - t_best
                t = t0_opt
                for n_fb in range(len(flybys)):
                    t = t + flybys[n_fb]["tof"]

                flybys = build_sequence(
                    J,
                    t,
                    flybys,
                    bodies,
                    min_tof,
                    max_tof,
                    tof_tries_nb,
                    t_max,
                    max_revs_lmbrt,
                    rin,
                    vin,
                    body_excluded=body_id_exclude,
                )

            # Batch des flybys
            for n_fb in range(len(flybys) - 1):
                # check si le segment n'est pas déjà solved
                if os.path.exists(save_path_json):
                    with open(save_path_json, "r") as f:
                        try:
                            results_json = json.load(f)
                        except json.JSONDecodeError:
                            results_json = []
                else:
                    results_json = []

                # Si pas déjà solved on le run
                if n_fb > int(len(results_json) - 1):

                    flyby = flybys[n_fb]
                    r1 = flyby["r2"]
                    v1 = flyby["v2"]
                    r1 = np.array(r1, dtype=float)
                    v1 = np.array(v1, dtype=float)

                    next_flyby = flybys[n_fb + 1]
                    r2 = next_flyby["r2"]
                    v2 = next_flyby["v2"]
                    r2 = np.array(r2, dtype=float)
                    v2 = np.array(v2, dtype=float)
                    tof = np.array(next_flyby["tof"], dtype=float)
                    v_dp_lmbrt = np.array(next_flyby["v1"], dtype=float)
                    vaf_fb = np.array(next_flyby["vout"], dtype=float)
                    guess = next_flyby["dv_left"]

                    conic = 0
                    g_test = np.array(guess)
                    if np.all(g_test == 0):
                        conic = 1

                    if conic == 0:
                        # Recalage erreur de pointage du au switch de methode de propagation
                        [tout, rout, vout] = propagate(
                            r1,
                            v_dp_lmbrt,
                            tof,
                            MU_ALTAIRA,
                            [np.pi / 2, np.pi / 2, np.pi / 2, np.pi / 2],
                            2,
                        )
                        err_pos = np.dot(rout[-1] - r2, rout[-1] - r2) / 10**10
                        err_vel = np.dot(vout[-1] - v2, vout[-1] - v2)
                        cost = err_pos + err_vel
                        print("cout du au switch de propagation :", cost)

                        print("vecteur delta à l'arrivé")
                        print(vout[-1] - v2)
                        ##      alpha0,beta0 = alpha_beta_from_u(r2, v2, vout[-1] - v2)
                        ##      print("in alpha / beta :", [alpha0,beta0])
                        ##      [alpha0, beta0] = alpha_beta_from_u(r1, vaf_fb, guess, eps=1e-12)
                        ##      print("in alpha / beta :", [alpha0,beta0])

                        # Création du paramétrage initial
                        N = 10
                        lb = -np.pi / 2
                        ub = np.pi / 2
                        bounds = [(lb, ub)] * N * 2
                        # initialisation des noeuds : vecteurs 1D longueur N
                        par_start_alpha = np.full((N,), np.pi / 2 - 0.1)
                        par_start_beta = np.full((N,), np.pi / 2 - 0.1)

                        # remplacer premier noeud par la solution lambert
                        # alpha0,beta0 = alpha_beta_from_u(r1, v1, guess)
                        # par_start_alpha[0] = alpha0
                        # par_start_beta[0]  = beta0
                        save_path_pkl = "res_one_shot.pkl"
                        par_start = np.concatenate((par_start_alpha, par_start_beta))

                        # si only refine on récupère les résultats ici
                        if os.path.exists(save_path_pkl):
                            print(
                                "🟢 Cached result found - skipping propagation error compensation."
                            )
                            with open(save_path_pkl, "rb") as f:
                                all_results = pickle.load(f)
                                res = all_results[i]
                            par_start = res.x
                            print("Loaded cached parameters.")

                        if only_refine == False:

                            # ---- Destruction du cache si présent ----
                            if os.path.exists(save_path_pkl):
                                os.remove(save_path_pkl)
                                print("Cache results intermediate cleaned")
                            all_results = {}
                            print("🆕 Nouveau cache créé.")

                            if 0 == 2:
                                continue
                            else:
                                # ---- Tentative unitaire initiale ----
                                res = minimize(
                                    error_lmbrt,
                                    par_start,
                                    args=(r1, v_dp_lmbrt, r2, v2, tof, MU_ALTAIRA, N),
                                    method="L-BFGS-B",
                                    bounds=bounds,
                                    options={
                                        "maxiter": 2000,
                                        "xatol": 1e-4,
                                        "fatol": 1e-6,
                                    },
                                )
                                print("One shot try gives a cost of :", res.fun)

                                # Si la tentative unitaire échoue → correction progressive
                                if res.fun > 1:
                                    print("Begin loop of propagation correction")

                                    # ---- Paramètres globaux ----
                                    N_interp = 10
                                    methods = [
                                        "L-BFGS-B",
                                        "Powell",
                                        "SLSQP",
                                    ]  #'Nelder-Mead'
                                    max_step_reductions = 3
                                    error_threshold = 100.0

                                    # ---- Préparation interpolation ----
                                    r_comp = np.linspace(rout[-1], r2, N_interp)
                                    v_comp = np.linspace(vout[-1], v2, N_interp)

                                    # ---- Boucle principale ----
                                    for i in range(N_interp):
                                        r_i = r_comp[i, :]
                                        v_i = v_comp[i, :]

                                        print(f"\n--- Step {i+1}/{N_interp} ---")
                                        print("Target error :", 1e-6)

                                        success = False
                                        step_scale = 1.0

                                        # Tentatives avec réduction progressive de la marche si échec
                                        for step_try in range(max_step_reductions):
                                            # On recalcule les cibles atténuées (on “revient en arrière”)
                                            r_target = rout[-1] + step_scale * (
                                                r_i - rout[-1]
                                            )
                                            v_target = vout[-1] + step_scale * (
                                                v_i - vout[-1]
                                            )

                                            for method in methods:
                                                print(
                                                    f"\nTrying method: {method}, step scale = {step_scale:.3f}"
                                                )

                                                res = minimize(
                                                    error_lmbrt,
                                                    par_start,
                                                    args=(
                                                        r1,
                                                        v_dp_lmbrt,
                                                        r_target,
                                                        v_target,
                                                        tof,
                                                        MU_ALTAIRA,
                                                        N,
                                                    ),
                                                    method=method,
                                                    bounds=bounds,
                                                    options={
                                                        "maxiter": 500,
                                                        "xatol": 1e-4,
                                                        "fatol": 1e-6,
                                                    },
                                                )

                                                cost = float(res.fun)
                                                print(f" → Cost = {cost:.4e}")

                                                if cost < error_threshold:
                                                    print(
                                                        f"✅ Success with {method} (cost={cost:.2f})"
                                                    )
                                                    par_start = res.x
                                                    success = True
                                                    break  # sortie de la boucle méthode

                                            if success:
                                                break  # sortie de la boucle de réduction de pas

                                            # Si toutes les méthodes échouent → réduction du pas spatial
                                            step_scale /= 2.0
                                            print(
                                                f"⚠️ All methods failed. Reducing step scale to {step_scale:.3f} and retrying..."
                                            )

                                        if not success:
                                            print(
                                                "❌ No convergence even after step reduction. Continuing to next interpolation point."
                                            )
                                            par_start = res.x
                                        print(
                                            f"Measured error at run end: {res.fun:.3e}"
                                        )
                                # Réussite one shot svg pour la suite
                                else:
                                    par_start = res.x

                                # ---- Sauvegarde finale du résultat (après correction si nécessaire) ----
                                all_results[n_fb] = res
                                with open(save_path, "wb") as f:
                                    pickle.dump(all_results, f)
                                print(
                                    f"resultat de l’arc {n_fb} sauvegardé ({len(all_results)} total)."
                                )

                            ###################  FIN RESOLUTION PHASE 1 #################3
                            # --- Phase 2 : résolution du problème low thrust ---
                            print("\n=== Begin Solar Sail Arc Solving ===")
                            print("Equivalent impulsive DV is :", guess)
                            # ---- Paramètres globaux ----
                            N_interp2 = 10
                            methods = ["L-BFGS-B", "Powell"]  #'Nelder-Mead'
                            max_subdivisions = 1
                            error_threshold = 100.0
                            factor = np.linspace(1, 0, N_interp2)

                            # ============================================================
                            # 🔧 Fonction récursive principale : solve_segment
                            # ============================================================
                            def solve_segment(
                                v_start, v_end, par_guess, bounds, depth=0
                            ):
                                """Résout un segment (v_start → v_end) de manière récursive avec multi-méthodes et réutilisation intelligente."""
                                print(
                                    f"\n➡ Solving subsegment depth={depth} "
                                    f"({np.linalg.norm(v_start):.3f}) → ({np.linalg.norm(v_end):.3f})"
                                )

                                best_res = None
                                best_cost = np.inf
                                current_guess = par_guess.copy()

                                # --- boucle multi-méthode avec propagation du meilleur x ---
                                for method in methods:

                                    try:
                                        res_try = minimize(
                                            error_lmbrt,
                                            current_guess,
                                            args=(
                                                r1,
                                                v_end,
                                                r2,
                                                v2,
                                                tof,
                                                MU_ALTAIRA,
                                                N,
                                            ),
                                            method=method,
                                            bounds=bounds,
                                            options={
                                                "disp": False,
                                                "maxiter": 1000,
                                                "xatol": 1e-3,
                                                "fatol": 1e-3,
                                            },
                                        )
                                        cost_try = float(res_try.fun)
                                        print(
                                            f"   {method:<12} → cost = {cost_try:.3e}"
                                        )

                                        if cost_try < best_cost:
                                            best_cost = cost_try
                                            best_res = res_try
                                            current_guess = (
                                                res_try.x.copy()
                                            )  # 🔑 On met à jour le guess pour la méthode suivante
                                            if best_cost < error_threshold:
                                                return current_guess

                                    except Exception as e:
                                        print(f"⚠ {method} failed: {e}")
                                        continue

                                # --- Décision ---
                                if best_res is not None and best_cost < error_threshold:
                                    print(
                                        f"✅ Subsegment converged (cost={best_cost:.3e})"
                                    )
                                    return best_res.x.copy()

                                # --- Si on atteint la profondeur max ---
                                if depth >= max_subdivisions:
                                    print(
                                        f"❌ Max subdivision reached at depth {depth}, keeping best result."
                                    )
                                    return (
                                        best_res.x.copy()
                                        if best_res is not None
                                        else par_guess.copy()
                                    )

                                # --- Sinon, subdivision récursive avec propagation de la meilleure solution ---
                                print(
                                    f"⚠ Subsegment failed (cost={best_cost:.2f}), subdividing..."
                                )
                                mid_v = 0.5 * (v_start + v_end)

                                # Utilise le meilleur x actuel pour lancer les sous-segments
                                par_mid = solve_segment(
                                    v_start, mid_v, current_guess, bounds, depth + 1
                                )
                                par_end = solve_segment(
                                    mid_v, v_end, par_mid, bounds, depth + 1
                                )

                                return par_end.copy()

                            # ============================================================
                            # 🔹 ONE SHOT GLOBAL (sans utilisation de "guess")
                            # ============================================================
                            print("One shot try begins")
                            ##  print("One shot try begins (randomized)")
                            ##  par_start, best_cost = random_start_optimization(
                            ##      error_func=error_lmbrt,
                            ##      par_start=par_start,
                            ##      args=(r1, vaf_fb, r2, v2, tof, MU_ALTAIRA, N),
                            ##      n_random=100,
                            ##      error_threshold=error_threshold
                            ##  )
                            lb = -np.pi / 2
                            ub = np.pi / 2
                            bounds = [(lb, ub)] * N * 2
                            res_global = minimize(
                                error_lmbrt,
                                par_start,
                                args=(r1, vaf_fb, r2, v2, tof, MU_ALTAIRA, N),
                                method="L-BFGS-B",
                                bounds=bounds,
                                options={
                                    "disp": True,
                                    "maxiter": 1,
                                    "xatol": 1e-3,
                                    "fatol": 1e-3,
                                },  ############################# DISABLED
                            )
                            cost = float(res_global.fun)
                            print(f"Global one-shot result: {cost:.3e}")

                            if cost < error_threshold:
                                print(
                                    "✅ Global one-shot succeeded — skipping interpolation phase."
                                )
                                par_start = res_global.x.copy()
                            else:
                                print(
                                    "⚠️ Global one-shot failed — starting adaptive interpolation."
                                )
                                v_prev = vaf_fb + guess * factor[0]

                                par_start_alpha = np.full((N,), np.pi / 2 - 0.1)
                                par_start_beta = np.full((N,), np.pi / 2 - 0.1)
                                par_start = np.concatenate(
                                    (par_start_alpha, par_start_beta)
                                )

                                par_global = par_start

                                for i in range(N_interp2):
                                    print(
                                        f"\n--- Solar sail step {i+1}/{N_interp2} ---"
                                    )
                                    v_target = vaf_fb + guess * factor[i]
                                    print(vaf_fb)
                                    print(vaf_fb + guess)
                                    print(v_dp_lmbrt)

                                    par_global = solve_segment(
                                        v_prev, v_target, par_global, bounds, depth=0
                                    )
                                    v_prev = v_target
                                    print(
                                        f"✅ Completed interpolation step {i+1}/{N_interp2}"
                                    )

                                par_start = par_global.copy()

                            # ============================================================
                            # 💾 Sauvegarde finale
                            # ============================================================
                            print("\n💾 Saving final optimized parameters...")
                            np.save("par_final.npy", par_start)
                            print("✅ Saved as 'par_final.npy'")
                        else:
                            par_start = np.load("par_final.npy")

                        # Cout avant raffinement
                        cost_bf_refine = error_lmbrt(
                            par_start, r1, vaf_fb, r2, v2, tof, MU_ALTAIRA, N
                        )
                        print(" Cost before refinement is : ", cost_bf_refine)

                        # === Phase finale de raffinement de précision ===
                        print("\n=== Final refinement phase (target 1e-6) ===")

                        ## RESOLUTION FINALE POUR ATTEINDRE LA TOL
                        # Objectif et tolérances
                        target_error = 1e-6
                        max_refine_iter = 10
                        max_ite_main_loop = 5
                        N_refine = N
                        methods_refine = ["L-BFGS-B", "Powell", "SLSQP"]  #'Nelder-Mead'
                        lb = -np.pi / 2
                        ub = np.pi / 2
                        bounds = [(lb, ub)] * N_refine * 2

                        best_res = res
                        best_res.fun = np.inf
                        # par_start = np.clip(par_start, -np.pi/2, np.pi/2)
                        prev_x = par_start

                        # === Initialisation ===
                        propagate_factor = np.linspace(0.9, 0.5, max_refine_iter)

                        # Meilleur résultat initial (grande valeur)
                        best_res = type("obj", (object,), {"fun": np.inf, "x": None})()
                        prev_x = None

                        # === Boucle principale de raffinement ===
                        for refine_round in range(max_refine_iter):
                            print(
                                f"\n🔹 Refinement round {refine_round+1}/{max_refine_iter}"
                            )
                            prop_fac = propagate_factor[refine_round]

                            # Propagation sur la dernière frac (modifiée dans la boucle for)
                            sol, rout, vout = propagate(
                                r1, vaf_fb, tof, MU_ALTAIRA, par_start, N
                            )
                            y = sol.sol(tof * prop_fac)
                            rout = y[0:3]
                            vout = y[3:6]

                            # Création du contrôle d'origine
                            alpha_spline, beta_spline = make_control_splines(
                                par_start, N, tof
                            )

                            # --- Pré-échantillonnage ---
                            N_cmd = 2000
                            # nombre d’échantillons de commande
                            t_cmd = np.linspace(0, tof, N_cmd)
                            alpha_tab = alpha_spline(t_cmd)
                            beta_tab = beta_spline(t_cmd)

                            tOUT, rout2, vout2 = propagate_sub_control(
                                r1,
                                vaf_fb,
                                tof,
                                MU_ALTAIRA,
                                np.linspace(0, 0, N * 2),
                                N,
                                alpha_tab,
                                beta_tab,
                                tof,
                                0,
                            )
                            rout2, vout2 = rout2[-1], vout2[-1]
                            err_pos = np.dot(rout2 - r2, rout2 - r2) / 10**10
                            err_vel = 0  # np.dot(vout2 - v2, vout2 - v2)
                            cost = err_pos + err_vel
                            print("Erreur controlée sur segment 2  :", cost)

                            # controle additionel
                            t_new = np.linspace(tof * prop_fac, tof, N)
                            alpha_step_sub = np.interp(t_new, t_cmd, alpha_tab)
                            beta_step_sub = np.interp(t_new, t_cmd, beta_tab)
                            # Chaque paramètre a une borne basse et haute → liste de tuples [(low1, high1), (low2, high2), ...]
                            bounds = []
                            for a, b in zip(alpha_step_sub, beta_step_sub):
                                # bounds.append((0,0))
                                # bounds.append((0,0))
                                bounds.append(
                                    (-np.pi / 2 - a, np.pi / 2 - a)
                                )  # borne pour alpha
                                bounds.append(
                                    (-np.pi / 2 - b, np.pi / 2 - b)
                                )  # borne pour beta)]
                            sub_par = np.linspace(0, 0, N * 2)
                            par_refine = sub_par

                            # Ajouter un p bruit aléatoire uniforme
                            # noise_level = 0.8
                            # par_refine = sub_par + np.random.uniform(-noise_level, noise_level, size=sub_par.shape)

                            # === Optimisation multi-méthode ===
                            for method in methods_refine:
                                print(
                                    f"→ Trying {method} with propagate_factor={prop_fac:.2f}"
                                )
                                res_try = minimize(
                                    error_lmbrt_sub_control,
                                    par_refine,
                                    args=(
                                        rout,
                                        vout,
                                        r2,
                                        v2,
                                        tof * (1 - prop_fac),
                                        MU_ALTAIRA,
                                        N,
                                        alpha_tab,
                                        beta_tab,
                                        tof,
                                        prop_fac,
                                    ),
                                    method=method,
                                    bounds=bounds,
                                    options={
                                        "maxiter": 5000,
                                        "xatol": 1e-8,
                                        "fatol": 1e-8,
                                    },
                                )

                                print(f"   cost = {res_try.fun:.3e}")

                                if res_try.fun < best_res.fun:
                                    best_res = res_try
                                    prev_x = res_try.x
                                    best_propag_factor = prop_fac
                                    print(f"✅ Improved to cost {res_try.fun:.3e}")
                                    tOUT, rout2, vout2 = propagate_sub_control(
                                        rout,
                                        vout,
                                        tof * (1 - prop_fac),
                                        MU_ALTAIRA,
                                        prev_x,
                                        N,
                                        alpha_tab,
                                        beta_tab,
                                        tof,
                                        prop_fac,
                                    )
                                    print("vecteur d'erreur en pos : ", r2 - rout2[-1])
                                    print("vecteur d'erreur en vel : ", v2 - vout2[-1])

                                if best_res.fun < target_error:
                                    print(
                                        "🎯 Target precision reached — stopping refinement."
                                    )
                                    break

                            if best_res.fun < target_error:
                                break  # plus besoin d’autres raffinements

                        res = best_res
                        print(f"Final refined cost: {res.fun:.3e}")

                        # Si pas d'ammélioration on prend les paramètres à 0
                        if best_res.fun > cost_bf_refine:
                            res.fun = cost_bf_refine
                            res.x = np.linspace(0, 0, N * 2)
                            best_propag_factor = 1

                        if res.fun > target_error:
                            print("no convergence stopping here")
                            break

                        # STEP 4 Passage au PMP

                        # ============ TBD ================== ##

                        #

                        # === Fin du traitement du flyby courant ===
                        print("\n=== Final optimization result ===")
                        print(res)

                    save_path_json = (
                        "results_flybys.json"  # lisible par Python facilement
                    )

                    # --- Version structurée pour relecture facile en Python ---
                    if conic == 0:
                        result_dict = {
                            "flyby_index": n_fb,
                            "flyby_total": len(flybys) - 1,
                            "cost": float(res.fun),
                            "best_propag_factor": float(best_propag_factor),
                            "par_start": par_start.tolist(),
                            "x": res.x.tolist(),
                        }
                    else:
                        result_dict = {
                            "flyby_index": n_fb,
                            "flyby_total": len(flybys) - 1,
                            "cost": 0,
                            "best_propag_factor": 0,
                            "par_start": 0,
                            "x": 0,
                        }

                    # Append ou création du JSON cumulatif
                    if os.path.exists(save_path_json):
                        with open(save_path_json, "r") as f:
                            try:
                                results_json = json.load(f)
                            except json.JSONDecodeError:
                                results_json = []
                    else:
                        results_json = []

                    results_json.append(result_dict)

                    with open(save_path_json, "w") as f:
                        json.dump(results_json, f, indent=2)

                    print(f"💾 Structured result saved to {save_path_json}")
                    if n_fb == len(flybys) - 2:
                        print(
                            "all segments solved ! Next step is writing GTOC13 submission file"
                        )
                        break

    # Ecriture du fichier de sorties final

    if Write_GTOC_output_file:
        output_table = []
        # Spline size
        N = 10
        # reouverture de init_pos

        data = np.load("save_state.npy", allow_pickle=True).item()

        pos_0_opt = data["r_init"]
        v_0_opt = data["v_init"]
        tp_first_pass_opt = data["t"]
        t_best = data["t_best"]
        t0_opt = tp_first_pass_opt - t_best
        body_aimed = 1

        ##    filename = "init_pos.csv"
        ##    with open(filename, mode="r", newline="") as f:
        ##      reader = csv.reader(f)
        ##      header = next(reader)
        ##      values = next(reader)
        ##
        ##    # convertir en float64
        ##    vals = np.array(values, dtype=np.float64)
        ##
        ##    score_over_time = vals[0]
        ##    t0_opt = vals[1]
        ##    tp_first_pass_opt = vals[2]
        ##    pos_0_opt = vals[3:6]
        ##    v_0_opt = vals[6:9]
        ##    rf_min_opt = vals[9:12]
        ##    vf_min_opt = vals[12:15]
        ##    body_aimed = vals[15]

        # Reouverture des fichier flyby et flyby results
        flybys = load_flybys_from_csv(filename="flybys.csv")

        with open("results_flybys.json") as f:
            data = json.load(f)

        for i in range(len(flybys)):

            if i == 0:
                conic = 1

            if i == 0:
                t = t0_opt
                r = pos_0_opt
                v = v_0_opt
                k = 1
            else:
                # Extraction des valeurs
                flyby_data = data[i - 1]
                flyby_index = flyby_data["flyby_index"]
                cost = flyby_data["cost"]
                prop_fac = flyby_data["best_propag_factor"]
                par_start = np.array(flyby_data["par_start"])
                prev_x = np.array(flyby_data["x"])

            flyby = flybys[i]
            r2 = flyby["r2"]
            v2 = flyby["v2"]
            body_id = int(flyby["body_id"])
            tof = flyby["tof"]

            dv_left = flyby["dv_left"]
            dv_test = np.array(dv_left)
            if np.all(dv_test == 0):
                conic = 1

            if i < len(flybys) - 1:
                next_flyby = flybys[i + 1]
                vout = next_flyby["vout"]
            else:
                # dernier flyby on prend n'importe lequel
                rend, vend_pl = bodies[body_id].eph((t + tof) / 86400)
                vout = np.array(
                    pk.fb_prop(
                        v2,
                        vend_pl,
                        bodies[body_id].safe_radius * 10,
                        3.1415 / 2,
                        bodies[body_id].mu_self,
                    )
                )

            if conic:
                output_table.append(
                    (k, 0, 0, t, np.array(r) / 1e3, np.array(v) / 1e3, 0)
                )  # PT INITIAL CONIQUE
                k = k + 1
                t = t + tof
                r = r2
                v = v2

                # Planet position at tof
                planet_init = bodies[body_id]
                rp, vp = planet_init.eph(t / 86400)

                rp0, vp0 = planet_init.eph(0)
                rp1, vp1 = pk.propagate_lagrangian(r0=rp0, v0=vp0, tof=t, mu=MU_ALTAIRA)

                ##        import matplotlib.pyplot as plt
                ##
                ##        fig = plt.figure()
                ##        ax = pk.orbit_plots.plot_planet(planet_init,tf=t, color='b')
                ##        pk.orbit_plots.plot_kepler(r0 = pos_0_opt, v0 = v_0_opt, tof =10000000000 , mu =MU_ALTAIRA, axes = ax, N=600000)
                ##        plt.show()
                print("l'ecriture est évaluée à :")
                print(r2)
                print(rp)
                print(t)

                # Patched Conics tolerance check
                rdiff = rp - r2
                if np.linalg.norm(rdiff) > 100:
                    print(
                        "planet rendezvous pos error is above tol :",
                        np.linalg.norm(rdiff),
                        "m",
                    )
                    # raise Exception('ai')
                vdiff = np.linalg.norm(v2 - vp) - np.linalg.norm(vout - vp)
                print(v2)
                # print(vf_min_opt)
                print(vout)
                if np.linalg.norm(vdiff) > 0.0001:
                    print(
                        "planet rendezvous velocity error is above tol :",
                        np.linalg.norm(vdiff),
                        "m/s",
                    )
                    raise Exception("ai")

                # Fin de l'arc conique
                output_table.append(
                    (k, 0, 0, t, np.array(r) / 1e3, np.array(v) / 1e3, 0)
                )
                k = k + 1

                # Début arc flyby
                output_table.append(
                    (
                        k,
                        body_id,
                        1,
                        t,
                        np.array(r) / 1e3,
                        np.array(v) / 1e3,
                        (v2 - vp) / 1e3,
                    )
                )
                k = k + 1
                output_table.append(
                    (
                        k,
                        body_id,
                        1,
                        t,
                        np.array(r) / 1e3,
                        np.array(vout) / 1e3,
                        (vout - vp) / 1e3,
                    )
                )
                k = k + 1
                v = vout
            else:

                # Reconstruction de la loi de contrôle sur l'arc
                alpha_spline, beta_spline = make_control_splines(par_start, N, tof)

                # --- Pré-échantillonnage ---
                N_cmd = 2000
                # nombre d’échantillons de commande
                t_cmd = np.linspace(0, tof, N_cmd)
                alpha_tab = alpha_spline(t_cmd)
                beta_tab = beta_spline(t_cmd)

                # Ajout de la loi raffinée sur segment additionel
                alpha_spline, beta_spline = make_control_splines(
                    prev_x, N, (tof * 1 - prop_fac)
                )
                # --- Pré-échantillonnage ---
                N_cmd = 2000
                # nombre d’échantillons de commande
                t_cmd = np.linspace(0, tof, N_cmd)
                alpha_tab2 = alpha_spline(t_cmd)
                beta_tab2 = beta_spline(t_cmd)

                # Propagation arc
                # Propagation sur la premire frac
                sol, rout, vout = propagate(r, v, tof, MU_ALTAIRA, par_start, N)

                for j in range(len(sol.t)):
                    ti = sol.t[j]
                    if ti < tof * prop_fac:
                        y = sol.sol(ti)
                        r = y[0:3]
                        v = y[3:6]
                        alpha = np.interp(ti, t_cmd, alpha_tab)
                        beta = np.interp(ti, t_cmd, beta_tab)
                        # On convertit alpha beta en inertiel
                        u = u_from_alpha_beta(r, v, alpha, beta)

                        output_table.append(
                            (k, 0, 1, t + ti, np.array(r) / 1e3, np.array(v) / 1e3, u)
                        )
                        k = k + 1
                    else:
                        # écriture doublée de la discontinuitée
                        ti = tof * prop_fac
                        y = sol.sol(tof * prop_fac)
                        r = y[0:3]
                        v = y[3:6]
                        alpha = np.interp(ti, t_cmd, alpha_tab)
                        beta = np.interp(ti, t_cmd, beta_tab)
                        # On convertit alpha beta en inertiel
                        u = u_from_alpha_beta(r, v, alpha, beta)
                        output_table.append(
                            (k, 0, 1, t + ti, np.array(r) / 1e3, np.array(v) / 1e3, u)
                        )
                        k = k + 1
                        # Mise à jour de la réérence temporelle
                        t = t + ti
                        break

                # Création du sub-control additionel
                alpha_tab_old = alpha_tab
                beta_tab_old = beta_tab

                alpha_spline, beta_spline = make_control_splines(
                    prev_x, N, tof * (1 - prop_fac)
                )

                # --- Pré-échantillonnage rapide ---
                N_cmd = 2000
                # nombre d’échantillons de commande
                t_cmd = np.linspace(0, tof * (1 - prop_fac), N_cmd)
                alpha_tab = alpha_spline(t_cmd)
                beta_tab = beta_spline(t_cmd)

                t_old_cmd = np.linspace(0, tof, N_cmd)
                t_cmd_reset = np.linspace(tof * prop_fac, tof, N_cmd)

                alpha_add = np.interp(t_cmd_reset, t_old_cmd, alpha_tab_old)
                beta_add = np.interp(t_cmd_reset, t_old_cmd, beta_tab_old)

                # somme du controle initial et du controle additionnel
                alpha_tab = alpha_tab + alpha_add
                beta_tab = beta_tab + beta_add

                # Propagation sur la deuxième portion
                sol, rout2, vout2 = propagate_sub_control(
                    r,
                    v,
                    tof * (1 - prop_fac),
                    MU_ALTAIRA,
                    prev_x,
                    N,
                    alpha_tab,
                    beta_tab,
                    tof,
                    prop_fac,
                )

                for j in range(len(sol.t)):
                    ti = sol.t[j]
                    y = sol.sol(ti)
                    r = y[0:3]
                    v = y[3:6]
                    alpha = np.interp(ti, t_cmd, alpha_tab)
                    beta = np.interp(ti, t_cmd, beta_tab)

                    # Vérification des bornes de contrôle
                    if alpha > np.pi / 2 + 0.0001 or alpha < -np.pi / 2 - 0.0001:
                        print(
                            "Le contrôle alpha est en dehors des bornes :", alpha, "rad"
                        )
                    if beta > np.pi / 2 + 0.0001 or beta < -np.pi / 2 - 0.0001:
                        print(
                            "Le contrôle beta est en dehors des bornes :", alpha, "rad"
                        )

                    # On convertit alpha beta en inertiel
                    u = u_from_alpha_beta(r, v, alpha, beta)
                    output_table.append(
                        (k, 0, 1, t + ti, np.array(r) / 1e3, np.array(v) / 1e3, u)
                    )
                    k = k + 1

                # mise à jour ref temporelle
                t = t + ti

                # Planet position at tof
                planet_init = planets[body_id]
                rp, vp = planet_init.eph(t / 86400)

                # Patched Conics tolerance check
                rdiff = rp - r2
                if np.linalg.norm(rdiff) > 100:
                    print(
                        "planet rendezvous pos error is above tol :",
                        np.linalg.norm(rdiff),
                        "m",
                    )
                    raise Exception("ai")
                vdiff = np.linalg.norm(v2 - vp) - np.linalg.norm(vout - vp)
                if np.linalg.norm(vdiff) > 0.0001:
                    print(
                        "planet rendezvous velocity error is above tol :",
                        np.linalg.norm(vdiff),
                        "m/s",
                    )
                    raise Exception("ai")

                # Début arc flyby
                output_table.append(
                    (
                        k,
                        body_id,
                        1,
                        t,
                        np.array(r) / 1e3,
                        np.array(v) / 1e3,
                        (v2 - vp) / 1e3,
                    )
                )
                k = k + 1
                output_table.append(
                    (
                        k,
                        body_id,
                        1,
                        t,
                        np.array(r) / 1e3,
                        np.array(vout) / 1e3,
                        (vout - vp) / 1e3,
                    )
                )
                k = k + 1
                v = vout

        write_gtoc13_solution("submission.txt", output_table)

    # svg des files dans un sous répertoire
    count = count + 1
    if count < 150:
        backup_current_folder()
    else:
        break
