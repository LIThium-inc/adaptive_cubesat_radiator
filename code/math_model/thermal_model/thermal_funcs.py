import math

import thermal_model.parameters as par #type: ignore


def T_hot_epsilon(epsilon_eff):
    A_eff = par.A_eff_at_max + par.c * (epsilon_eff - par.epsilon_eff_max)
    Q_in = (par.q * (par.S_rad * A_eff + par.C_)
            + par.Q_hot_refl + par.Q_earth + par.Q_hot_internal)
    return math.pow(Q_in / (par.sigma * (par.S_rad * epsilon_eff + par.C)), 0.25)


def T_cold_epsilon(epsilon_eff):
    return math.pow(par.G / (par.sigma * (par.S_rad * epsilon_eff + par.C)), 0.25)


