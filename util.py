import numpy as np
from typing import Union


# Define equations
# Davis Equation - Models the resistance forces
# Road Load Force
def road_load_force(a: float, b: float, c: float, v: float) -> float:  # v: [m/s]
    f_v: float = a + b * v + c * np.power(v, 2)  # [N]
    return f_v


# Road Load Power
def road_load_power(v: float, f_v: float) -> float:  # v: [m/s], f_v: [N]
    p_v: float = f_v * v  # [W]
    return p_v  # [W]


# Gradient effect
# Gradient force function
def gradient_force(
    alpha: float, a: float, b: float, c: float, v: float, m: float, g: float
) -> float:  # alpha: [degrees], v: [m/s]
    f_v = road_load_force(a, b, c, v)
    f_r: float = (m * g * np.sin(np.radians(alpha))) + f_v
    return f_r  # [N]


# Gradient power function
def gradient_power(
    alpha: float, a: float, b: float, c: float, v: float, m: float, g: float
) -> float:
    p_r = gradient_force(alpha, a, b, c, v, m, g) * v
    return p_r  # [W]


# Motor Characteristic Curve
# Radians to rpm
def rads_to_rpm(rad: float) -> float:
    return rad * 30 / (np.pi)


# Rated nominal speed
def rated_nominal_speed(p_m_r: float, t_m_r: float):  # p_m_r: [W], t_m_r: [Nm]
    w_m_r = p_m_r / (t_m_r)
    return w_m_r  # [rad/s]


# Constant Torque Mode
def constant_torque_mode(
    w_m: Union[float, np.ndarray], w_m_r: float, t_m_r: float, p_m_r: float
) -> Union[float, np.ndarray]:  # w_m: [rad/s], w_m_r: [rad/s], p_m_r: [W], t_m_r: [Nm]
    t_m = np.where(
        w_m <= w_m_r,
        t_m_r,  # constant toque mode
        p_m_r / w_m,  # constant power mode
    )
    return t_m  # [Nm]


# Constant Power Mode
def constant_power_mode(
    w_m: Union[float, np.ndarray], w_m_r: float, t_m_r: float, p_m_r: float
) -> Union[float, np.ndarray]:  # w_m: [rad/s], w_m_r: [rad/s], p_m_r: [W], t_m_r: [Nm]
    p_m = np.where(
        w_m <= w_m_r,
        t_m_r * w_m,  # constant toque mode
        p_m_r,  # constant power mode
    )
    return p_m  # [W]


# Braking distance
def braking_distance(v_a: float, v_b: float, a_max_brake: float):
    b_d = 0.5 * ((v_b**2) - (v_a**2)) / a_max_brake
    return b_d


# Acceleration
# Traction a > 0 - No mechanical brakes
def a_direct(
    alpha: int,  # degrees
    A: float,  # N
    B: float,  # Ns/m
    C: float,  # Ns2/m2
    v: float,  # m/s
    t_m: float,  # Nm
    m: float,  # kg
    g: float,  # m/s2
    eta_d: float,  # -
    r_w: float,  # m
    tau_g: float,  # -
    j_m: float,  # kgm2
    j_w: float,  # kgm2
) -> float:  # m/s2
    f_r_tot = gradient_force(alpha, A, B, C, v, m, g)
    # Motor force
    f_m = (eta_d * t_m) / (r_w * tau_g)

    return (f_m - f_r_tot) / (
        m + eta_d * j_m / ((r_w * tau_g) ** 2) + 4 * j_w / (r_w**2)
    )


# Braking with motor only
def a_reverse(
    alpha: int,  # Degrees
    A: float,  # N
    B: float,  # Ns/m
    C: float,  # Ns2/m2
    v: float,  # m/s
    t_m: float,  # Nm
    m: float,  # kg
    g: float,  # m/s2
    eta_r: float,  # -
    j_m: float,  # kgm2 Hp
    r_w: float,  # m
    tau_g: float,  # -
    j_w: float,  # kgm2 Hp
) -> float:  # m/s2
    f_r_tot = gradient_force(alpha, A, B, C, v, m, g)

    return (t_m / (r_w * tau_g) - eta_r * f_r_tot) / (
        eta_r * m + j_m / ((r_w * tau_g) ** 2) + eta_r * 4 * j_w / (r_w**2)
    )


# Traction force
def friction_force(
    alpha: int,  # Degrees
    m: float,  # kg
    g: float,  # m/s2
    mu: float,  # -
    v: float,  # m/s
    in_traction_wheels: int,  # -
    total_wheels: int,  # -
) -> float:
    n_traction = m * g * np.cos(alpha) * in_traction_wheels / total_wheels
    mu_v = mu / (1 + 0.01 * v)

    return mu_v * n_traction
