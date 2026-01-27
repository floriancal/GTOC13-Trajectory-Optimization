# Import needed modules
import numpy as np
from collections import defaultdict
import csv
import pykep as pk
import matplotlib.pyplot as plt
import orbital
from utilities import *
from orbital import *
from scipy.optimize import minimize
import os
import pickle
import json
import random
import time 
from constants import MU_ALTAIRA, AU, t_max, C, A, m, r0

print("Welcome to GTOC13 optimization program")

# Here we are in the main loop
# Hyperparams
tof_tries_nb = (
    100  # Size of the times_of_flight vector to parse in the sequence finding
)
min_tof = 10 * 86400  # Minimum tof between two planets [s]
max_tof = 50 * 86400 * 365.25  # Maximum tof between two planets [s]
max_revs_lmbrt = 100  # In the Lambert solving algorithm there is 2N+1 solutions with N being the number of revolutions

max_period_aimed = 10 * 365.25 * 86400 # In first_legs_search this is the orbital period to aim for  

V0_min = 3000  # This is the initial Min and Max Velocities in [m/s]
V0_max = 50000

# What to execute
initial_point_search = False  # Not to be runned each time, this builds a table of velocity/position and times vectors at first body encounter as a function of the initial state of the problem
first_legs_search = True # Wether to compute or reuse the choice of the two firsts conic legs
sequence_search = True # Wether to run the flybys sequence search or not
only_conics = True # If running a sequence search and orbital reduction legs choose to consider only conic arcs or not (use the sail or not)

# For now the solve_solar_sail_arcs capabilty is not available arcs must be conics, nevertheless the structure has been prepared
solve_solar_sail_arcs = False  # Wether to run the solving of the sail arcs (must at least be runned once after a sequence search even in only_conics for Xcheck)
only_refine = False  # Wether to only run the refinement of the solar sail solving of controlled arcs (the solving process is divided in two steps)
Write_GTOC_output_file = True  # Wether to prepare the GTOC format solution file or not
Draw_mission = True # at the end of the run draw the trajectory on a 3D plot


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
        print("For now this part of the script has no end and will parse all V0 then offset lambda 0 by 10° modulo 2 * pi then iterate again forever, it should be stopped by the user when 'init_pos.csv' is considered big enough ")
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
                    args=(V_space[i], rp, MU_ALTAIRA),
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

                    # Check ALTAIRA proximity constraint (we do not store shield state data)
                    invalid, shield_state_not_used = shield_state(False, rf, vf, rf_min, vf_min, tof_init)

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
        
        if first_legs_search == True:
            print("Begining orbital period reduction legs..")
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
            
            while True:
                flybys = []
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
                # Proximity constraint calculation for first leg 
                invalid, shield_burned = shield_state(shield_burned, r_init, v_init, r1_fb1, v1_fb1, tof_fb1)
                
                if not invalid :
   
                    if t + tof2 < t_max:
                        
                        # Planet velocity at t  
                        _, vpfby = bodies[body_aimed].eph(t  / 86400)                        
                        
                        new_flyby = {
                            "body_id": body_aimed,
                            "r_hat": r1_fb1 / np.linalg.norm(r1_fb1),
                            "Vinf": np.linalg.norm(np.array(v1_fb1) - np.array(vpfby)),
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
                                        
                        # Second flyby
                        # ALTAIRA proximity constraint 
                        invalid, shield_burned = shield_state(shield_burned, r1_fb1, v1, r2, v2, tof2)
                        
                        if not invalid : 
                            
                            # Planet velocity at t + tof 2 
                            _, vpfby = bodies[body_id_fb2].eph((t + tof2) / 86400)
                            
                            new_flyby = {
                                "body_id": body_id_fb2,
                                "r_hat": r2_body_j / np.linalg.norm(r2_body_j),
                                "Vinf": np.linalg.norm(np.array(v2) - np.array(vpfby)),
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
                            
                            
                            # Now goal is to evaluate orbital period to find a reasonable one, if not for now iterate 
                            flybys = build_sequence(
                                J,
                                t,
                                flybys,
                                bodies,
                                10, # MIN TOF
                                max_period_aimed, # MAX TOF
                                100,
                                t_max,
                                max_revs_lmbrt,
                                planets, # which ones to use for the sequence here aonly planets
                                orbital_period_search = True,
                                only_conics = only_conics
                            )
                            
                            # At the last flyby 
                            r_last = flybys[-1]["r2"]
                            v_last = flybys[-1]["v2"]
                            
                            # Orbital period ? 
                            el_orb_search = pk.ic2par(r_last,v_last, MU_ALTAIRA)
                            # Mandatory direct shot
                            if el_orb_search[1] > 1:
                                T = np.inf              
                            else:         
                                T = 2*np.pi * np.sqrt(el_orb_search[0]**3/MU_ALTAIRA)
                            
                            print("orbital period at end of scan n is", T)
                            print("orbital eccentricity is", el_orb_search[1])
                            print("target is ", max_period_aimed)
                            
                            # Compute actual mission time
                            # t is already at end of second leg t0 + tof1 + tof2
                            for fb in flybys[2:]:
                                t += fb["tof"]
                            print("tnow is :", t)
                            
                            # Save t for next runs in addition to the flybys dict
                            print("save")
                            with open("t.pkl", "wb") as f:
                                pickle.dump(t, f) 
                                
                            if T < max_period_aimed:
                                break
        
                    else:
                        print("First two segments are already > t_max : ", t / 86400 / 365.25)
                        break
                    
  
                    
            else:
                flybys = load_flybys_from_csv(filename="flyby_first_two_legs.csv")
                J = objective(flybys)


                
        ### -------------- SEQUENCE SEARCH ----------------###      
        # Initial state is the state of the last flyby 
        with open("t.pkl", "rb") as f:
            t = pickle.load(f)
        print("reopen t", t)
        
        print("End orbital period reduction legs, beginning optimal sequence search")
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
            planets, # which ones to use for the sequence here all bodies(now we try just planets)
            only_conics = only_conics
        )
        save_flybys_to_csv(flybys, filename="flybys_first_row.csv")


    # Or we reload past solution
    else:
        flybys = load_flybys_from_csv(filename="flybys.csv")
        J = 0 # Undefined for now but need to be initialized

    if solve_solar_sail_arcs:
        # Begin solar sail arc (in case of only_conic check that the arcs are really conic)
        print("Begining sail control arcs solving")
        save_path = "results_flybys.txt"
        save_path_json = "results_flybys.json"

        # --- Clean files
        if os.path.exists(save_path):
            os.remove(save_path)
            print(f" Fichier {save_path} supprimé en début de run.")
        if os.path.exists(save_path_json):
            os.remove(save_path_json)
            print(f" Fichier {save_path} supprimé en début de run.")
        
        # In case of solar sail arcs if an arc cannot be solved the sequence search is rebuilded again from the failed arc
        for depth_nb in range(sequence_search_depth):
            # Re-run sequence search excluding last sol
            if depth_nb > 0:
                # the list is sliced to the failed arc
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
                
                # Sequence is rebuilded
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
                    planets,
                    body_excluded=body_id_exclude,
                )

            # Batch des flybys
            for n_fb in range(len(flybys) - 1):
                # We check if the segment has already been solved 
                if os.path.exists(save_path_json):
                    with open(save_path_json, "r") as f:
                        try:
                            results_json = json.load(f)
                        except json.JSONDecodeError:
                            results_json = []
                else:
                    results_json = []

                # Otherwise we run it 
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
                        raise Exception("An arc is not conic, but the solve_solar_sail_arcs function is still TBD")
                        # TBD solve_solar_sail_arcs() 

                    # Build the file 
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

                    # Json creation
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

                    print(f" Structured result saved to {save_path_json}")
                    if n_fb == len(flybys) - 2:
                        print(
                            "all segments solved ! Next step is writing GTOC13 submission file"
                        )
                        break
            if n_fb == len(flybys) - 2:
                break

    # Writing of the submission file
    if Write_GTOC_output_file or Draw_mission:
        shield_burned = False
        
        if Draw_mission:
            print('drawing !')
            
            sc_handle = None

            # lets check trajectory visually at each run 
            planets = load_bodies_from_csv("gtoc13_planets.csv")

            plt.ion()  
            ax = None
            for i in range(len(planets)-1):
                 planet_plot = planets[i+1]
                 ax = pk.orbit_plots.plot_planet(planet_plot, color='b', axes=ax)
            ax.set_autoscale_on(False)

            plt.show()
            plt.pause(0.001)

        output_table = []
        # Spline size
        N = 10
        data = np.load("save_state.npy", allow_pickle=True).item()

        pos_0_opt = data["r_init"]
        v_0_opt = data["v_init"]
        tp_first_pass_opt = data["t"]
        t_best = data["t_best"]
        t0_opt = tp_first_pass_opt - t_best
        body_aimed = 1

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
                # Values extraction
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
                # Last flyby we simulate an non existent planet rdv
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
                
                # Check Altaira proximity constraints
                invalid, shield_burned = shield_state(shield_burned, r, v, r2, v2, tof)
                if invalid:
                    rmin, _ = min_radius_calc(r, v, r2, v2, tof)
                    print('Min dist observed in [AU] : ', rmin/AU)
                    
                    raise Exception('Invalid perihelion detected')
                    

                output_table.append(
                    (k, 0, 0, t, np.array(r) / 1e3, np.array(v) / 1e3, 0)
                )  # Initial point
                k = k + 1
                t = t + tof
                
                def round_sig(x, sig=16):
                    return float(f"{x:.{sig}g}")
                
                r = r2
                v = v2

                # Planet position at tof
                planet_init = bodies[body_id]
                rp, vp = planet_init.eph(t / 86400)

                rp0, vp0 = planet_init.eph(0)
                
                
                if Draw_mission:
                    
                    # Actual visualization
                    xlim = ax.get_xlim3d()
                    ylim = ax.get_ylim3d()
                    zlim = ax.get_zlim3d()
                    
                    # Plot spacecraft
                    pk.orbit_plots.plot_kepler(r, v, -tof, MU_ALTAIRA, axes = ax, N=6000, color='g')
                    print("At flyby : ", i+1, "Score is :", objective(flybys[0:i+1]))

                 
                    # Restore vue 
                    ax.set_xlim3d(xlim)
                    ax.set_ylim3d(ylim)
                    ax.set_zlim3d(zlim)

                    # Suppress past SC point 
                    if sc_handle is not None:
                        sc_handle.remove()

                    # Add new SC point 
                    sc_handle = ax.scatter(
                        r[0], r[1], r[2],
                        color='r',
                        s=40,
                        label='SC @ t'
                    )
    
                    while True: 
                        key = plt.waitforbuttonpress()
                        if key: 
                            break

                # Patched Conics tolerance check
                rdiff = rp - r2
                if np.linalg.norm(rdiff) > 100:
                    print(
                        "planet rendezvous pos error is above tol :",
                        np.linalg.norm(rdiff),
                        "m",
                    )
                    raise Exception('ai')
                #print('rdiff', np.linalg.norm(rdiff))
                #print('rt',rt)
                #print('rfile propag with file data ', rfile)
                #print('Check ', r_check)
                vdiff = np.linalg.norm(v2 - vp) - np.linalg.norm(vout - vp)
                print('check')
                if np.linalg.norm(vdiff) > 0.0001:
                    print(
                        "planet rendezvous velocity error is above tol :",
                        np.linalg.norm(vdiff),
                        "m/s",
                    )
                    raise Exception("ai")

                # End conic arc 
                output_table.append(
                    (k, 0, 0, t, np.array(r) / 1e3, np.array(v) / 1e3, 0)
                )
                k = k + 1

                # Begin flyby arc
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
                ### ----- For now this mode is unsupported --- ###
                ### ------------------------------------------ ###
                ### ------------------------------------------ ###
                # Rebuild the control law
                alpha_spline, beta_spline = make_control_splines(par_start, N, tof)

                # --- Re-use the same point frequency TBU---
                N_cmd = 2000
                t_cmd = np.linspace(0, tof, N_cmd)
                alpha_tab = alpha_spline(t_cmd)
                beta_tab = beta_spline(t_cmd)

                # Add refined law (final law = global law + refined law)
                alpha_spline, beta_spline = make_control_splines(
                    prev_x, N, (tof * 1 - prop_fac)
                )
 
                N_cmd = 2000
                t_cmd = np.linspace(0, tof, N_cmd)
                alpha_tab2 = alpha_spline(t_cmd)
                beta_tab2 = beta_spline(t_cmd)

                # Propagation arc
                # Propagation on the first part (before refinement)
                sol, rout, vout = propagate(r, v, tof, MU_ALTAIRA, par_start, N)

                for j in range(len(sol.t)):
                    ti = sol.t[j]
                    if ti < tof * prop_fac:
                        y = sol.sol(ti)
                        r = y[0:3]
                        v = y[3:6]
                        alpha = np.interp(ti, t_cmd, alpha_tab)
                        beta = np.interp(ti, t_cmd, beta_tab)
                        # alpha beta are converted to inertial frame
                        u = u_from_alpha_beta(r, v, alpha, beta)

                        output_table.append(
                            (k, 0, 1, t + ti, np.array(r) / 1e3, np.array(v) / 1e3, u)
                        )
                        k = k + 1
                    else:
                        # we need to double up the point where the discntinuity exist (to respect the submission format)
                        ti = tof * prop_fac
                        y = sol.sol(tof * prop_fac)
                        r = y[0:3]
                        v = y[3:6]
                        alpha = np.interp(ti, t_cmd, alpha_tab)
                        beta = np.interp(ti, t_cmd, beta_tab)
                        u = u_from_alpha_beta(r, v, alpha, beta)
                        output_table.append(
                            (k, 0, 1, t + ti, np.array(r) / 1e3, np.array(v) / 1e3, u)
                        )
                        k = k + 1
                        # Update temporal reference
                        t = t + ti
                        break

                # Additional sub-control
                alpha_tab_old = alpha_tab
                beta_tab_old = beta_tab

                alpha_spline, beta_spline = make_control_splines(
                    prev_x, N, tof * (1 - prop_fac)
                )

                N_cmd = 2000
                t_cmd = np.linspace(0, tof * (1 - prop_fac), N_cmd)
                alpha_tab = alpha_spline(t_cmd)
                beta_tab = beta_spline(t_cmd)

                t_old_cmd = np.linspace(0, tof, N_cmd)
                t_cmd_reset = np.linspace(tof * prop_fac, tof, N_cmd)

                alpha_add = np.interp(t_cmd_reset, t_old_cmd, alpha_tab_old)
                beta_add = np.interp(t_cmd_reset, t_old_cmd, beta_tab_old)

                # Initial control + sub-control
                alpha_tab = alpha_tab + alpha_add
                beta_tab = beta_tab + beta_add

                # Propagate the second fraction
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

                    # Verification of control bounds
                    if alpha > np.pi / 2 + 0.0001 or alpha < -np.pi / 2 - 0.0001:
                        print(
                            "Alpha is outside control bounds :", alpha, "rad"
                        )
                    if beta > np.pi / 2 + 0.0001 or beta < -np.pi / 2 - 0.0001:
                        print(
                            "Beta is outside control bounds :", alpha, "rad"
                        )

                    # Alpha / Beta is converted to inertial 
                    u = u_from_alpha_beta(r, v, alpha, beta)
                    output_table.append(
                        (k, 0, 1, t + ti, np.array(r) / 1e3, np.array(v) / 1e3, u)
                    )
                    k = k + 1

                # Update temporal reference
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

                # Flyby arc begin
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
    
    # svg of files in subdirectory 
    count = count + 1
    if count < 150:
        backup_current_folder()
    else:
        break
        
    # If we only have writing or visualize run once
    if not initial_point_search and not first_legs_search and not sequence_search and not solve_solar_sail_arcs:
        break
if Draw_mission:
    plt.show(block = True)



# test propag
#print('r',r)
#print('v',v)

#rx=[-29919574138.20000076293945312, 33899323.20443549007177353, -0.00020453717028339]
#vx= [44.25375530199427487, 0.00000000000000000, 0.00000000000000000]
#tofx=676964354.22120845317840576 - 8465369.55306994915008545,
#rx = [round_sig(x, 11) for x in rx]
#vx = [round_sig(x, 11) for x in vx]
##tofx = round_sig(tof, 11)
#print(tofx)

#rt, vt = pk.propagate_lagrangian(r0=r, v0=v, tof=tof, mu=MU_ALTAIRA)
#rfile,vfile = pk.propagate_lagrangian(r0=rx, v0=vx, tof=tofx, mu=139348062043.343)  


#from poliastro.twobody.orbit import Orbit
#from poliastro.twobody.propagation import propagate
#from astropy import units as u
#from astropy.time import TimeDelta
#from poliastro.bodies import Body
#from poliastro.twobody.propagation import cowell

#rx = r * u.m
#vx = v * u.m / u.s
#tofi = tof * u.s
#Altaira = Body(parent=None, k=MU_ALTAIRA * u.m**3 / u.s**2, name="Altaira")

#orb = Orbit.from_vectors(Altaira, rx, vx)
#orb_f = propagate(orb, TimeDelta(tofi), method=cowell,
#rtol=1e-11)

#r_check = orb_f.r.to(u.m).value
#v_check = orb_f.v.to(u.m/u.s).value
