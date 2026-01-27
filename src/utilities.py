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
from orbital import *

DEG2RAD = np.pi / 180


# ------------------------------------------------------------
# GTOC13 – Building the complete catalog of bodies of the system Altaira
# ------------------------------------------------------------
def load_bodies_from_csv(filename, mu_default=10.0):
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

                if body_id >= 1000:
                    radius = 100

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


def backup_current_folder():
    """
    Create a timestamped backup of selected files in the current directory.

    This function creates a subdirectory named with the current date and
    time, then copies all files in the current working directory whose
    extensions match a predefined list into this backup folder.

    Currently supported file extensions are: `.csv`, `.txt`, and `.npy`.

    The function does not recurse into subdirectories and only copies
    regular files located in the current directory.

    Notes
    -----
    - The backup directory is named using the format:
      ``backup_YYYYMMDD_HHMMSS``.
    - Existing backup directories with the same name are allowed
      (timestamp resolution is one second).
    - File metadata is preserved according to the default behavior of
      `shutil.copy`.
    - Progress information is printed to standard output.
    """

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


def run_with_timeout(func, args=(), timeout=5):
    """
    Execute a function with a time limit using a separate thread.

    The function `func` is executed in a background thread and is allowed
    to run for at most `timeout` seconds. If the execution exceeds this
    duration, the function returns None. If the wrapped function raises
    an exception, it is re-raised in the caller context.

    Parameters
    ----------
    func : callable
        Function to execute.
    args : tuple, optional
        Positional arguments passed to `func`.
        Default is an empty tuple.
    timeout : float, optional
        Maximum execution time in seconds.
        Default is 5 seconds.

    Returns
    -------
    result : any or None
        Return value of `func` if it completes within the time limit.
        Returns None if the execution times out.

    Raises
    ------
    Exception
        Any exception raised by `func` during execution.

    Notes
    -----
    - The execution is performed in a separate thread; the underlying
      function is not forcibly terminated if a timeout occurs.
    - This utility is suitable for I/O-bound or cooperative tasks, but
      may not reliably interrupt CPU-bound computations.
    """
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


def save_flybys_to_csv(flybys, filename="flybys.csv"):
    """
    Save a list of flyby dictionaries to a CSV file.

    Parameters
    ----------
    flybys : list of dict
        List of flyby dictionaries. Each dictionary may contain entries such as:
        ``'body_id'``, ``'r_hat'``, ``'Vinf'``, ``'is_science'``, ``'r2'``,
        ``'v2'``, ``'tof'``, etc.
    filename : str, optional
        Output CSV file name. Default is ``"flybys.csv"``.
    """
    if len(flybys) == 0:
        print("pas de flyby a sauvegarder")
        return

    all_keys = set()
    for fb in flybys:
        all_keys.update(fb.keys())
    fieldnames = list(all_keys)

    #  Write csv file
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
    Load flybys from a CSV file with automatic type parsing.

    The function attempts to convert CSV values to appropriate Python types:
    - Vectors: NumPy arrays from formats like [1 2 3], [1;2;3], [1, 2, 3], (1, 2, 3)
    - Booleans: True / False
    - Numbers: int or float
    - Otherwise: raw string

    Parameters
    ----------
    filename : str, optional
        CSV file to load. Default is "flybys.csv".

    Returns
    -------
    flybys : list of dict
        List of flyby dictionaries reconstructed from the CSV file.
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

                    # --- VECTORS ---
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

                    # --- OTHER (string brut) ---
                    fb[key] = val

                flybys.append(fb)

    except FileNotFoundError:
        print(f" Fichier '{filename}' introuvable.")
    return flybys
