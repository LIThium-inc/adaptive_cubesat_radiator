import math

import matplotlib.pyplot as plt
import numpy as np

import thermal_model.parameters as par  # type: ignore


def T_hot_S(S_rad, epsilon_eff):
    """T_hot для заданной площади радиатора S_rad (м^2).

    Та же формула, что в T_hot_epsilon, но S_rad передаётся аргументом,
    а не берётся из par.S_rad.
    """
    A_eff = par.A_eff_at_max + par.c * (epsilon_eff - par.epsilon_eff_max)
    Q_in = (par.q * (S_rad * A_eff + par.C_)
            + par.Q_hot_refl + par.Q_earth + par.Q_hot_internal)
    return math.pow(Q_in / (par.sigma * (S_rad * epsilon_eff + par.C)), 0.25)


EPS_CLOSED = par.epsilon_eff_min  # радиатор "закрыт" (минимальное достижимое ε_eff)
EPS_OPEN = par.epsilon_eff_max    # радиатор "открыт" (максимальное достижимое ε_eff)

# Площадь радиатора: от 1 см^2 до 5 м^2 (логарифмическая сетка)
S_values = np.logspace(-4, math.log10(5), 400)

T_closed = np.array([T_hot_S(S, EPS_CLOSED) for S in S_values])
T_open = np.array([T_hot_S(S, EPS_OPEN) for S in S_values])
delta_T = T_closed - T_open

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)

# --- График 1: температуры в двух состояниях ---
ax1.plot(S_values, T_closed - 273.15, label=f"T_hot, ε = {EPS_CLOSED} (закрыт)", linewidth=2)
ax1.plot(S_values, T_open - 273.15, label=f"T_hot, ε = {EPS_OPEN} (открыт)", linewidth=2)
ax1.set_ylabel("Температура, °C")
ax1.set_title("Горячий режим: температура и ΔT в зависимости от площади радиатора")
ax1.grid(True, which="both", linestyle="--", alpha=0.5)
ax1.legend()

# --- График 2: ΔT ---
ax2.plot(S_values, delta_T, color="tab:red", linewidth=2,
         label="ΔT = T_hot(ε_закр) − T_hot(ε_откр)")
ax2.set_xlabel("Площадь радиатора S_rad, м²")
ax2.set_ylabel("ΔT, К")
ax2.set_xscale("log")
ax2.grid(True, which="both", linestyle="--", alpha=0.5)
ax2.legend()

# Для ориентира: площадь всей поверхности 3U (~0.34 м^2)
for ax in (ax1, ax2):
    ax.axvline(0.34, color="gray", linestyle=":", linewidth=1)
ax2.text(0.34, ax2.get_ylim()[1] * 0.95, " вся поверхность 3U",
         color="gray", va="top")

plt.tight_layout()
plt.show()