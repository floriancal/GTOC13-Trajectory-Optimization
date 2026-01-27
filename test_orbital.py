from orbital import *
import numpy as np
import matplotlib.pyplot as plt
# Domaine de Vinf : 0 à 100 000 m/s
Vinf = np.linspace(0, 100_000, 2000)

# Évaluation
F = F_velocity(Vinf)

# Plot
plt.figure()
plt.plot(Vinf, F)
plt.xlabel("Vinf [m/s]")
plt.ylabel("F_velocity")
plt.title("F_velocity en fonction de Vinf (0–100 km/s)")
plt.grid(True)
plt.show()



# --- Exemple du document ---
r1 = np.array([0.0, 1.0, 0.0])  # r̂_k,1

theta_deg = np.linspace(0, 360, 2000)
theta_rad = np.radians(theta_deg)

S = []

for th in theta_rad:
    r2 = np.array([np.cos(th), np.sin(th), 0.0])
    S.append(S_seasonal([r1, r2]))

S = np.array(S)

# --- Plot ---
plt.figure()
plt.plot(theta_deg, S)
plt.xlabel("θ [deg]")
plt.ylabel("S")
plt.title("Seasonal penalty term S (2 flybys)")
plt.grid(True)
plt.show()


flybys = load_flybys_from_csv(filename="flybys.csv")
print(objective(flybys))


new_flyby = {
                            "body_id": 10,
                            "r_hat": [1,0,0],
                            "Vinf": 10000,
                            "is_science": True,
                        }
                        
flybys = [new_flyby]
print(objective(flybys))