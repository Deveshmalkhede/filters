"""
Kalman filter: 1-D constant-velocity tracker.

A cart moves along a line at roughly constant velocity. We have a noisy
POSITION sensor only -- velocity is never measured. The filter recovers it
anyway, from the correlation structure that builds up in P.

State:        x = [position, velocity]^T          (n = 2)
Measurement:  z = position + noise                (m = 1)

Every line of the filter below maps to one of the five equations.
"""

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# ----------------------------------------------------------------------
# 1. Scenario
# ----------------------------------------------------------------------
dt = 0.1                 # timestep [s]
T = 20.0                 # total duration [s]
N = int(T / dt)

true_x0 = 0.0            # true initial position [m]
true_v = 2.0             # true velocity [m/s]

sigma_a = 0.20           # process noise: std of unmodeled accel [m/s^2]
sigma_z = 1.50           # sensor noise: std of position measurement [m]

# ----------------------------------------------------------------------
# 2. Model matrices
# ----------------------------------------------------------------------

# F -- state transition. Constant velocity: p_k = p + v*dt,  v_k = v
F = np.array([[1.0, dt],
              [0.0, 1.0]])

# H -- measurement matrix. We see position only, so it SELECTS position.
#      m=1 < n=2: fewer observations than states. This is the interesting case.
H = np.array([[1.0, 0.0]])

# Q -- process noise covariance.
#      Discrete white-noise acceleration model: an unknown accel a ~ N(0, sigma_a^2)
#      acts over dt, perturbing position by a*dt^2/2 and velocity by a*dt.
#      Q = G G^T sigma_a^2  with  G = [dt^2/2, dt]^T
G = np.array([[0.5 * dt**2],
              [dt]])
Q = G @ G.T * sigma_a**2

# R -- measurement noise covariance (1x1 here)
R = np.array([[sigma_z**2]])

# ----------------------------------------------------------------------
# 3. Ground truth + synthetic measurements
# ----------------------------------------------------------------------
x_true = np.zeros((N, 2))
x_true[0] = [true_x0, true_v]

for k in range(1, N):
    # propagate truth with REAL process noise injected
    a = rng.normal(0.0, sigma_a)              # the unmodeled acceleration
    w = G.flatten() * a
    x_true[k] = F @ x_true[k - 1] + w

# noisy position measurements
z_meas = x_true[:, 0] + rng.normal(0.0, sigma_z, N)

# ----------------------------------------------------------------------
# 4. The filter
# ----------------------------------------------------------------------

# Initial belief. Deliberately WRONG and uncertain, to watch it converge:
# we claim position 5 m and velocity 0, with big covariance to admit we
# are only guessing.
x_hat = np.array([5.0, 0.0])
P = np.diag([10.0**2, 5.0**2])

I = np.eye(2)

# logs
est = np.zeros((N, 2))
std = np.zeros((N, 2))       # sqrt of diagonal of P -> 1-sigma per state
gain = np.zeros((N, 2))      # Kalman gain, one entry per state
innov = np.zeros(N)          # innovation z - H x^-
innov_std = np.zeros(N)      # sqrt(S), the expected innovation spread

for k in range(N):
    # ---- PREDICT ----------------------------------------------------
    # no control input here, so the B u term drops out
    x_hat = F @ x_hat                                  # (1)  x^- = F x
    P = F @ P @ F.T + Q                                # (2)  P^- = F P F^T + Q

    # ---- UPDATE -----------------------------------------------------
    z = np.array([z_meas[k]])

    y = z - H @ x_hat                                  #      innovation
    S = H @ P @ H.T + R                                #      innovation covariance
    K = P @ H.T @ np.linalg.inv(S)                     # (3)  K = P^- H^T S^-1

    x_hat = x_hat + (K @ y)                            # (4)  x = x^- + K y
    P = (I - K @ H) @ P                                # (5)  P = (I - K H) P^-

    est[k] = x_hat
    std[k] = np.sqrt(np.diag(P))
    gain[k] = K.flatten()
    innov[k] = y[0]
    innov_std[k] = np.sqrt(S[0, 0])

t = np.arange(N) * dt

# ----------------------------------------------------------------------
# 5. Results
# ----------------------------------------------------------------------
rmse_meas = np.sqrt(np.mean((z_meas - x_true[:, 0]) ** 2))
rmse_pos = np.sqrt(np.mean((est[:, 0] - x_true[:, 0]) ** 2))
rmse_vel = np.sqrt(np.mean((est[N // 4:, 1] - x_true[N // 4:, 1]) ** 2))

print(f"raw measurement RMSE (position) : {rmse_meas:.3f} m")
print(f"filtered        RMSE (position) : {rmse_pos:.3f} m")
print(f"  -> noise reduced by a factor of {rmse_meas / rmse_pos:.1f}x")
print()
print(f"velocity RMSE after settling    : {rmse_vel:.3f} m/s   (never measured!)")
print(f"final velocity estimate         : {est[-1, 1]:.3f} m/s")
print(f"true  velocity at end           : {x_true[-1, 1]:.3f} m/s")
print()
print(f"steady-state Kalman gain        : K = [{gain[-1,0]:.4f}, {gain[-1,1]:.4f}]")
print(f"steady-state position 1-sigma   : {std[-1,0]:.3f} m  (sensor is {sigma_z} m)")

# ----------------------------------------------------------------------
# 6. Plots
# ----------------------------------------------------------------------
fig, ax = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("Kalman filter — 1-D constant-velocity tracker (position sensor only)",
             fontsize=14, fontweight="bold")

# -- position
a0 = ax[0, 0]
a0.plot(t, z_meas, ".", color="0.65", ms=4, label="measurements")
a0.plot(t, x_true[:, 0], "k-", lw=2, label="truth")
a0.plot(t, est[:, 0], "C3-", lw=1.8, label="KF estimate")
a0.fill_between(t, est[:, 0] - 2 * std[:, 0], est[:, 0] + 2 * std[:, 0],
                color="C3", alpha=0.20, label="±2σ")
a0.set_title("Position — signal pulled out of the noise")
a0.set_xlabel("time [s]"); a0.set_ylabel("position [m]")
a0.legend(loc="upper left"); a0.grid(alpha=0.3)

# -- velocity (never measured)
a1 = ax[0, 1]
a1.plot(t, x_true[:, 1], "k-", lw=2, label="true velocity")
a1.plot(t, est[:, 1], "C0-", lw=1.8, label="KF estimate")
a1.fill_between(t, est[:, 1] - 2 * std[:, 1], est[:, 1] + 2 * std[:, 1],
                color="C0", alpha=0.20, label="±2σ")
a1.axhline(0, color="0.8", lw=0.8)
a1.set_title("Velocity — INFERRED, no sensor ever measured it")
a1.set_xlabel("time [s]"); a1.set_ylabel("velocity [m/s]")
a1.legend(loc="lower right"); a1.grid(alpha=0.3)

# -- Kalman gain
a2 = ax[1, 0]
a2.plot(t, gain[:, 0], "C2-", lw=1.8, label="K[0]  (→ position)")
a2.plot(t, gain[:, 1], "C4-", lw=1.8, label="K[1]  (→ velocity)")
a2.set_title("Kalman gain — starts trusting the sensor, settles into a balance")
a2.set_xlabel("time [s]"); a2.set_ylabel("gain")
a2.legend(); a2.grid(alpha=0.3)

# -- uncertainty
a3 = ax[1, 1]
a3.plot(t, std[:, 0], "C3-", lw=1.8, label="position 1σ")
a3.plot(t, std[:, 1], "C0-", lw=1.8, label="velocity 1σ")
a3.axhline(sigma_z, color="0.5", ls="--", lw=1.2, label=f"raw sensor 1σ = {sigma_z} m")
a3.set_title("Uncertainty — collapses, then holds steady state")
a3.set_xlabel("time [s]"); a3.set_ylabel("standard deviation")
a3.set_yscale("log")
a3.legend(); a3.grid(alpha=0.3, which="both")

plt.tight_layout()
plt.savefig("kf_1d_tracker.png", dpi=130)
print("\nsaved plot -> kf_1d_tracker.png")
plt.show()
