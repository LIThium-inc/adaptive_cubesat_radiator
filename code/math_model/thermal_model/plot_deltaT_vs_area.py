import math

import matplotlib.pyplot as plt
import numpy as np

import thermal_model.parameters as par  # type: ignore
import thermal_model.thermal_funcs as tf  # type: ignore

EPS_MIN = par.epsilon_eff_min   # радиатор "закрыт" (минимальное достижимое ε_eff)
EPS_MAX = par.epsilon_eff_max   # радиатор "открыт" (максимальное достижимое ε_eff)

# Площадь радиатора, м^2 (логарифмическая сетка)
S = np.logspace(-4, math.log10(0.5), 400)

# Характерные площади одной грани 3U: торец 10x10 см и боковая грань 10x30 см
S_MARKS = {0.01: "торец", 0.03: "боковая грань"}

SIDE_STYLE = {"sun": ("-", "Солнце"), "earth": ("--", "Земля")}


def to_C(T_kelvin):
    return T_kelvin - 273.15


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)

# --- График 1: T_hot в закрытом и открытом состоянии ---
for side, (ls, side_name) in SIDE_STYLE.items():
    ax1.plot(S, to_C(tf.T_hot_epsilon(EPS_MIN, S, side)), ls, color="tab:blue", linewidth=2,
             label=f"ε = {EPS_MIN:.2f} (закрыт), радиатор на стороне: {side_name}")
    ax1.plot(S, to_C(tf.T_hot_epsilon(EPS_MAX, S, side)), ls, color="tab:orange", linewidth=2,
             label=f"ε = {EPS_MAX:.2f} (открыт), радиатор на стороне: {side_name}")
ax1.set_ylabel("T_hot, °C")
ax1.set_title("Горячий режим: температура в крайних состояниях радиатора")
ax1.legend(fontsize=8)

# --- График 2: разность температур между крайними состояниями ---
for side, (ls, side_name) in SIDE_STYLE.items():
    dT_hot = tf.T_hot_epsilon(EPS_MIN, S, side) - tf.T_hot_epsilon(EPS_MAX, S, side)
    dT_cold = tf.T_cold_epsilon(EPS_MIN, S, side) - tf.T_cold_epsilon(EPS_MAX, S, side)
    ax2.plot(S, dT_hot, ls, color="tab:red", linewidth=2,
             label=f"горячий режим, сторона: {side_name}")
    ax2.plot(S, dT_cold, ls, color="tab:blue", linewidth=2,
             label=f"холодный режим, сторона: {side_name}")
ax2.axhline(0, color="black", linewidth=0.8)
ax2.set_xscale("log")
ax2.set_xlabel("Площадь радиатора S_rad, м²")
ax2.set_ylabel("ΔT = T(ε_закр) − T(ε_откр), К")
ax2.set_title("Разность температур между закрытым и открытым состоянием "
              "(знак минус: открытый радиатор теплее)")
ax2.legend(fontsize=8)

for ax in (ax1, ax2):
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    for s_mark in S_MARKS:
        ax.axvline(s_mark, color="gray", linestyle=":", linewidth=1)
for s_mark, label in S_MARKS.items():
    ax2.text(s_mark, 0.98, " " + label, transform=ax2.get_xaxis_transform(),
             rotation=90, va="top", ha="right", fontsize=8, color="gray")

print(f"Диапазон ε_eff: {EPS_MIN:.3f} … {EPS_MAX:.3f}")
for s_mark, label in S_MARKS.items():
    for side, (_, side_name) in SIDE_STYLE.items():
        d_hot = float(tf.T_hot_epsilon(EPS_MIN, s_mark, side) - tf.T_hot_epsilon(EPS_MAX, s_mark, side))
        d_cold = float(tf.T_cold_epsilon(EPS_MIN, s_mark, side) - tf.T_cold_epsilon(EPS_MAX, s_mark, side))
        print(f"S = {s_mark} м² ({label}), сторона {side_name}: "
              f"ΔT_hot = {d_hot:.1f} К, ΔT_cold = {d_cold:.1f} К")

plt.tight_layout()
plt.show()