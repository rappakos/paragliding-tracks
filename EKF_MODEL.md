# EKF Model — Thermal Trigger-Point Estimation (time-based)

Extended Kalman Filter that fits a stationary, wind-leaning thermal plume to a
manually selected climb segment of a paraglider IGC track, then extrapolates the
plume centerline down to the ground (DEM intersection) to locate the trigger point.

The filter marches in **IGC sample time** `t`. Height is a state; the
stationary-plume assumption enters only through how the center advances with height
(`dC/dt = s·(w − w_sink)`).

---

## 1. Coordinate frame

Local metric frame (UTM or ENU): `x` east, `y` north, `z` up, all in meters.
Angles in radians, speeds in m/s.

## 2. State vector (8 states)

```
x = [ C_x, C_y, s_x, s_y, φ, R, w, h ]ᵀ
```

| # | symbol | meaning | unit |
|---|--------|---------|------|
| 0 | `C_x`  | plume centerline east position at current height | m |
| 1 | `C_y`  | plume centerline north position at current height | m |
| 2 | `s_x`  | lean slope `dC_x/dh` (east) | — |
| 3 | `s_y`  | lean slope `dC_y/dh` (north) | — |
| 4 | `φ`    | pilot orbital phase | rad |
| 5 | `R`    | circle radius | m |
| 6 | `w`    | core updraft speed | m/s |
| 7 | `h`    | pilot height | m |

Pilot ground position is **not** a state; it is the output of the measurement
function: `X = C_x + R·cos φ`, `Y = C_y + R·sin φ`, `Z = h`.

## 3. Known inputs / parameters

| symbol | meaning | typical |
|--------|---------|---------|
| `U(h) = (U_x, U_y)` | openmeteo wind interpolated to height `h` (control) | — |
| `v_tan` | air-relative tangential speed | 9 m/s |
| `w_sink` | air-relative sink rate | 1 m/s |
| `σ` | circling direction (+1 / −1) | ±1 |

`U(h)` is obtained by (linear) interpolation of the openmeteo wind over the
pressure-level heights; `dU/dh` is the slope of that interpolation at the current
height (needed for `H3`).

---

## 4. Process model (continuous)

```
Ċ_x = s_x · (w − w_sink)
Ċ_y = s_y · (w − w_sink)
ṡ_x = 0                  (+ Q_s : lean random walk)
ṡ_y = 0                  (+ Q_s)
φ̇  = σ · v_tan / R
Ṙ  = 0                  (+ Q_R)
ẇ  = 0                  (+ Q_w : lets updraft vary with height)
ḣ  = w − w_sink
```

### Process Jacobian `F = ∂f/∂x` (8×8, nonzero entries)

```
∂Ċ_x/∂s_x = (w − w_sink)     ∂Ċ_x/∂w = s_x
∂Ċ_y/∂s_y = (w − w_sink)     ∂Ċ_y/∂w = s_y
∂φ̇ /∂R   = −σ · v_tan / R²
∂ḣ /∂w    = 1
```

All other entries are 0. Rows for `s_x, s_y, R, w` are pure random walk (zero `F`
rows, driven only by `Q`).

### Discretization (Euler; RK4 optional)

```
x⁻      = x + f(x)·Δt
Φ       = I + F·Δt
P⁻      = Φ · P · Φᵀ + Q·|Δt|
```

`Q = diag(q_C, q_C, q_s, q_s, q_φ, q_R, q_w, q_h)` — tune `q_s` (lean wander),
`q_R`, `q_w` as the main knobs; keep `q_C`, `q_φ`, `q_h` small.

> Reverse-time run: initialize at the **top** of the track (best-centered) and
> integrate downward with `Δt < 0`, so `ḣ < 0`. Use `|Δt|` for the `Q` term.

---

## 5. Measurement models & Jacobians

Column order everywhere: `[ C_x, C_y, s_x, s_y, φ, R, w, h ]`.
Apply the three updates **sequentially** each step (different noise scales →
avoids an ill-conditioned stacked `R`).

### H1 — GPS 3D position (per sample)

```
z1   = [X_gps, Y_gps, Z_gps]ᵀ
h1(x)= [ C_x + R·cos φ ,
         C_y + R·sin φ ,
         h            ]
```

```
        C_x  C_y  s_x  s_y     φ         R      w   h
H1 =  [  1    0    0    0   −R·sinφ    cosφ    0   0     # ∂X
         0    1    0    0    R·cosφ    sinφ    0   0     # ∂Y
         0    0    0    0     0         0      0   1 ]   # ∂Z
```

`R1 = diag(σ_xy², σ_xy², σ_z²)`.

### H2 — vario (per sample; observes `w`)

```
z2   = dz/dt              (from pressure-altitude difference)
h2(x)= w − w_sink
```

```
        C_x  C_y  s_x  s_y   φ   R   w   h
H2 =  [  0    0    0    0    0   0   1   0 ]
```

`R2 = σ_vario²` — small (baro rate is precise).

### H3 — advection pseudo-measurement (per sample; regularizes lean toward U/w)

```
z3   = [0, 0]ᵀ
g(x) = [ s_x·w − U_x(h) ,
         s_y·w − U_y(h) ]
innovation = z3 − g(x) = −g(x)
```

```
        C_x  C_y  s_x  s_y   φ   R    w      h
H3 =  [  0    0    w    0    0   0   s_x   −dU_x/dh     # ∂g_x
         0    0    0    w    0   0   s_y   −dU_y/dh ]   # ∂g_y
```

`R3 = diag(r_adv, r_adv)` — the trust knob: small → lean follows wind, large →
lean free to follow the circle-centroid drift in the data.

---

## 6. Pseudo-code

### Prediction

```
function predict(x, P, dt):
    # f(x)
    climb = x.w - w_sink
    f = zeros(8)
    f[C_x] = x.s_x * climb
    f[C_y] = x.s_y * climb
    f[phi] = sigma * v_tan / x.R
    f[h]   = climb
    # (f[s_x]=f[s_y]=f[R]=f[w]=0)

    x_pred = x + f * dt
    x_pred.phi = wrap_to_pi(x_pred.phi)

    # F (continuous)
    F = zeros(8,8)
    F[C_x, s_x] = climb;   F[C_x, w] = x.s_x
    F[C_y, s_y] = climb;   F[C_y, w] = x.s_y
    F[phi, R]   = -sigma * v_tan / x.R**2
    F[h,   w]   = 1

    Phi = I(8) + F * dt
    P_pred = Phi @ P @ Phi.T + Q * abs(dt)
    return x_pred, P_pred
```

### Update (generic EKF, called once per measurement)

```
function update(x, P, z, h_pred, H, Rm):
    y = z - h_pred                      # innovation (for H3: h_pred=g(x), z=0)
    S = H @ P @ H.T + Rm
    K = P @ H.T @ inv(S)
    x = x + K @ y
    x.phi = wrap_to_pi(x.phi)
    # Joseph form for numerical stability
    A = I(8) - K @ H
    P = A @ P @ A.T + K @ Rm @ K.T
    return x, P
```

### One filter step

```
function step(x, P, meas, dt):
    x, P = predict(x, P, dt)

    # 1) GPS
    z1  = [meas.X, meas.Y, meas.Z]
    hp1 = [x.C_x + x.R*cos(x.phi), x.C_y + x.R*sin(x.phi), x.h]
    x, P = update(x, P, z1, hp1, H1(x), R1)

    # 2) vario
    z2  = meas.dz_dt
    hp2 = x.w - w_sink
    x, P = update(x, P, z2, hp2, H2, R2)

    # 3) advection constraint
    Ux, Uy   = wind(x.h);  dUx, dUy = wind_slope(x.h)
    z3  = [0, 0]
    hp3 = [x.s_x*x.w - Ux, x.s_y*x.w - Uy]
    x, P = update(x, P, z3, hp3, H3(x, dUx, dUy), R3)

    return x, P
```

### Driver (reverse-time, top → bottom, then smooth)

```
function run(track, x0, P0):
    order = sort_by_time_descending(track)      # start at top
    xs, Ps = [], []
    x, P = x0, P0
    for k in 1..len(order):
        dt = order[k].t - order[k-1].t          # negative
        x, P = step(x, P, order[k], dt)
        xs.append(x); Ps.append(P)
    return RTS_smoother(xs, Ps)                  # optional but recommended
```

## 7. Initialization (top of track)

Fit a circle to the last 1–2 turns:
`(C_x, C_y)` = centroid, `R` = fitted radius, `φ` from the last fix,
`σ` from turn direction; `h`, `w` from the top; `s = U(h)/w`.
Seed `P0` small on `C, R, φ`, larger on `s, w`.

## 8. Ground trigger

After smoothing you have `C(h)`, `s(h)`. Extrapolate below the lowest fix with
`dC/dh = s(h) = U(h)/w(h)` (taper `w` toward the surface) and solve the implicit
intersection with terrain by bisection:

```
find h_g such that  h_g = DEM( C_x(h_g), C_y(h_g) )
```

Propagate the smoothed covariance through the extrapolation and the DEM
interpolation → report the trigger as an uncertainty ellipse, not a point.
