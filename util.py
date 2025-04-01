import numpy as np


# Define equations
# Davis Equation - Models the resistance forces
# Road Load Force
def road_load_force(a, b, c, v):  # v: [m/s]
    f_v = a + b * v + c * np.power(v, 2)  # [N]
    return f_v


# Road Load Power
def road_load_power(v, f_v):  # v: [m/s], f_v: [N]
    p_v = f_v * v  # [W]
    return p_v  # [W]


# Gradient effect
# Gradient force function
def gradient_force(alpha, a, b, c, v, m, g):  # alpha: [degrees], v: [m/s]
    f_v = road_load_force(a, b, c, v)
    f_r = (m * g * np.sin(np.radians(alpha))) + f_v
    return f_r  # [N]


# Gradient power function
def gradient_power(alpha, v):
    p_r = gradient_force(alpha, v) * v
    return p_r  # [W]


# Motor Characteristic Curve
# Radians to rpm
def rads_to_rpm(rad):
    return rad * 30 / (np.pi)


# Rated nominal speed
def rated_nominal_speed(p_m_r, t_m_r):  # p_m_r: [W], t_m_r: [Nm]
    w_m_r = p_m_r / (t_m_r)
    return w_m_r  # [rad/s]


# Constant Torque Mode
def constant_torque_mode(
    w_m, w_m_r, t_m_r, p_m_r
):  # w_m: [rad/s], w_m_r: [rad/s], p_m_r: [W], t_m_r: [Nm]
    t_m = np.where(
        w_m <= w_m_r,
        t_m_r,  # constant toque mode
        p_m_r / w_m,  # constant power mode
    )
    return t_m  # [Nm]


# Constant Power Mode
def constant_power_mode(
    w_m, w_m_r, t_m_r, p_m_r
):  # w_m: [rad/s], w_m_r: [rad/s], p_m_r: [W], t_m_r: [Nm]
    p_m = np.where(
        w_m <= w_m_r,
        t_m_r * w_m,  # constant toque mode
        p_m_r,  # constant power mode
    )
    return p_m  # [W]


# 