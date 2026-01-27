def solve_solar_sail_arcs()		
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
            "Cached result found - skipping propagation error compensation."
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
        print("Nouveau cache créé.")

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
                                    f" Success with {method} (cost={cost:.2f})"
                                )
                                par_start = res.x
                                success = True
                                break  # sortie de la boucle méthode

                        if success:
                            break  # sortie de la boucle de réduction de pas

                        # Si toutes les méthodes échouent → réduction du pas spatial
                        step_scale /= 2.0
                        print(
                            f"All methods failed. Reducing step scale to {step_scale:.3f} and retrying..."
                        )

                    if not success:
                        print(
                            "No convergence even after step reduction. Continuing to next interpolation point."
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
        # Fonction récursive principale : solve_segment
        # ============================================================
        def solve_segment(
            v_start, v_end, par_guess, bounds, depth=0
        ):
            """Résout un segment (v_start → v_end) de manière récursive avec multi-méthodes et réutilisation intelligente."""
            print(
                f"\n Solving subsegment depth={depth} "
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
                        )  # On met à jour le guess pour la méthode suivante
                        if best_cost < error_threshold:
                            return current_guess

                except Exception as e:
                    print(f"⚠ {method} failed: {e}")
                    continue

            # --- Décision ---
            if best_res is not None and best_cost < error_threshold:
                print(
                    f" Subsegment converged (cost={best_cost:.3e})"
                )
                return best_res.x.copy()

            # --- Si on atteint la profondeur max ---
            if depth >= max_subdivisions:
                print(
                    f" Max subdivision reached at depth {depth}, keeping best result."
                )
                return (
                    best_res.x.copy()
                    if best_res is not None
                    else par_guess.copy()
                )

            # --- Sinon, subdivision récursive avec propagation de la meilleure solution ---
            print(
                f"Subsegment failed (cost={best_cost:.2f}), subdividing..."
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
        #  ONE SHOT GLOBAL (sans utilisation de "guess")
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
                "Global one-shot succeeded — skipping interpolation phase."
            )
            par_start = res_global.x.copy()
        else:
            print(
                "Global one-shot failed — starting adaptive interpolation."
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
                    f"Completed interpolation step {i+1}/{N_interp2}"
                )

            par_start = par_global.copy()

        # ============================================================
        # Sauvegarde finale
        # ============================================================
        print("\n Saving final optimized parameters...")
        np.save("par_final.npy", par_start)
        print("Saved as 'par_final.npy'")
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
            f"\n Refinement round {refine_round+1}/{max_refine_iter}"
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
                print(f" Improved to cost {res_try.fun:.3e}")
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
                    " Target precision reached — stopping refinement."
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

return # TBD