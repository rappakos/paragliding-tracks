# Paragliding Thermal Core Analysis: EKF Plume Model

This document outlines the mathematical foundation for identifying a thermal's geographic trigger point using an Extended Kalman Filter (EKF). Rather than tracking a moving bubble of air over time, this model takes an Eulerian approach: the thermal is treated as a stationary, time-independent spatial plume $\mathbf{C}(Z)$. The paraglider is modeled as a probe flying upwards (and spiraling) through this leaning structure.

To maximize stability, the filter processes the manually selected IGC segment **backwards** (top-down), anchoring the initial state in the clean, established orbits at the top of the thermal and pushing through the noisy, chaotic entry phase below.

---

## 1. System Definitions

### The State Vector

At any discrete time step $k$, the state vector $\mathbf{x}_k$ tracks the 2D absolute position of the thermal core at the glider's current altitude, and the 2D absolute position of the paraglider:

$$\mathbf{x}_k = \begin{bmatrix} c_{x,k} \\ c_{y,k} \\ X_k \\ Y_k \end{bmatrix}$$

### The Control Input

The core advection is driven by known external variables: the horizontal wind velocity $\mathbf{v}_w(Z_k)$ from Open-Meteo, the glider's measured vertical speed $\dot{Z}_k$, and the local thermal updraft $w_{th}(Z_k) = \dot{Z}_k - v_{sink}$.

The control input $\mathbf{u}_k$ defines the lateral shift of the core $\Delta \mathbf{C}_k$ between time steps due to the altitude change:

$$\mathbf{u}_k = \begin{bmatrix} \Delta c_{x,k} \\ \Delta c_{y,k} \end{bmatrix} = \begin{bmatrix} \frac{v_{wx}(Z_k)}{w_{th}(Z_k)} \dot{Z}_k \Delta t \\ \frac{v_{wy}(Z_k)}{w_{th}(Z_k)} \dot{Z}_k \Delta t \end{bmatrix}$$

*Note on top-down processing: Because the filter iterates backwards through the IGC array, $\Delta t = t_{k-1} - t_k$ will be negative, naturally unwinding the upward drift and the pilot's rotation.*

---

## 2. Process Model & Prediction Step

The state prediction $\hat{\mathbf{x}}_{k|k-1} = f(\hat{\mathbf{x}}_{k-1|k-1}, \mathbf{u}_{k-1})$ updates the core position using the control input and updates the glider position using a tangential rotational model.

Let the relative position be $x_r = X_{k-1} - c_{x,k-1}$ and $y_r = Y_{k-1} - c_{y,k-1}$, turn direction $d \in \{-1, 1\}$, tangential velocity $v_t \approx 9 \text{ m/s}$, and orbit radius $r = \sqrt{x_r^2 + y_r^2}$.

**State Prediction:**


$$\hat{\mathbf{x}}_{k|k-1} = \begin{bmatrix} c_{x,k-1} + \Delta c_{x,k-1} \\ c_{y,k-1} + \Delta c_{y,k-1} \\ X_{k-1} + \Delta c_{x,k-1} - d \frac{v_t}{r} y_r \Delta t \\ Y_{k-1} + \Delta c_{y,k-1} + d \frac{v_t}{r} x_r \Delta t \end{bmatrix}$$

**Covariance Prediction:**


$$\mathbf{P}_{k|k-1} = \mathbf{F}_{k-1} \mathbf{P}_{k-1|k-1} \mathbf{F}_{k-1}^T + \mathbf{Q}$$

### The Jacobian Matrix ($\mathbf{F}$)

Because the kinematics are non-linear, we linearize around the current estimate using the Jacobian $\mathbf{F} = \frac{\partial f}{\partial \mathbf{x}}$.

$$\mathbf{F}_{k-1} = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ \frac{\partial X_k}{\partial c_x} & \frac{\partial X_k}{\partial c_y} & \frac{\partial X_k}{\partial X} & \frac{\partial X_k}{\partial Y} \\ \frac{\partial Y_k}{\partial c_x} & \frac{\partial Y_k}{\partial c_y} & \frac{\partial Y_k}{\partial X} & \frac{\partial Y_k}{\partial Y} \end{bmatrix}$$

Where the partial derivatives are:

* $\frac{\partial X_k}{\partial X} = 1 + d \cdot v_t \frac{x_r y_r}{r^3} \Delta t$
* $\frac{\partial X_k}{\partial Y} = - d \cdot v_t \frac{x_r^2}{r^3} \Delta t$
* $\frac{\partial X_k}{\partial c_x} = 1 - \frac{\partial X_k}{\partial X}$
* $\frac{\partial X_k}{\partial c_y} = - \frac{\partial X_k}{\partial Y}$
* $\frac{\partial Y_k}{\partial X} = d \cdot v_t \frac{y_r^2}{r^3} \Delta t$
* $\frac{\partial Y_k}{\partial Y} = 1 - d \cdot v_t \frac{x_r y_r}{r^3} \Delta t$
* $\frac{\partial Y_k}{\partial c_x} = - \frac{\partial Y_k}{\partial X}$
* $\frac{\partial Y_k}{\partial c_y} = 1 - \frac{\partial Y_k}{\partial Y}$

---

## 3. Measurement Model & Update Step

The measurement $\mathbf{z}_k$ is the absolute GPS coordinate from the IGC track. This linear relationship allows for standard Kalman update equations.

**Measurement Vector and Matrix:**


$$\mathbf{z}_k = \begin{bmatrix} X_{GPS, k} \\ Y_{GPS, k} \end{bmatrix}$$

$$\mathbf{H} = \begin{bmatrix} 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}$$

**Update Equations:**

1. **Innovation:** $\mathbf{y}_k = \mathbf{z}_k - \mathbf{H} \hat{\mathbf{x}}_{k|k-1}$
2. **Innovation Covariance:** $\mathbf{S}_k = \mathbf{H} \mathbf{P}_{k|k-1} \mathbf{H}^T + \mathbf{R}$
3. **Kalman Gain:** $\mathbf{K}_k = \mathbf{P}_{k|k-1} \mathbf{H}^T \mathbf{S}_k^{-1}$
4. **State Update:** $\hat{\mathbf{x}}_{k|k} = \hat{\mathbf{x}}_{k|k-1} + \mathbf{K}_k \mathbf{y}_k$
5. **Covariance Update:** $\mathbf{P}_{k|k} = (\mathbf{I} - \mathbf{K}_k \mathbf{H}) \mathbf{P}_{k|k-1}$

---

## 4. Filter Initialization (Top-Down Anchor)

To initialize the filter prior to the reverse pass:

1. Extract the last 15-20 seconds of the user-selected IGC segment (representing the final, established top-level orbit).
2. Set the initial core state $[c_{x,N}, c_{y,N}]^T$ to the mean $X$ and $Y$ of this orbit.
3. Set the initial glider state $[X_N, Y_N]^T$ to the exact final GPS point.
4. Initialize $\mathbf{P}$ with low variance values for the core to heavily trust this top-level anchor point.

---

## 5. Extrapolation to Surface Trigger Point

*(Details on dynamic integration from $Z_{entry}$ to ground elevation utilizing interpolated $w_{th}$ boundaries to be added here.)*