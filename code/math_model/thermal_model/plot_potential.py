import matplotlib.pyplot as plt
import numpy as np

import thermal_model.parameters as par  # type: ignore
import thermal_model.thermal_funcs as tf  # type: ignore

# ====== Настройки ======
T_MAX_C = 50.0                  # максимально допустимая температура, °C
T_MIN_LIST_C = [0.0, -20.0]     # варианты минимально допустимой температуры, °C

# Площади радиатора, м^2: наш случай и боковая грань 3U (10x34 см) для сравнения.
# Чтобы построить только наш случай, оставьте в списке один элемент: [par.S_rad]
S_LIST = [par.S_rad, 0.034]

N_DX = 120      # число точек по размаху Δε
N_EPS = 120     # число шагов по ε внутри диапазона [ε_min, ε_max]

# Верхняя граница диапазона ε_eff фиксирована (радиатор "открыт"), размах меняется за счёт ε_min.
# Линия A(ε) берётся из parameters.py (A_eff_at_max и c) и продлевается вниз по ε.
EPS_MAX = par.epsilon_eff_max
DX_MAX = EPS_MAX - par.epsilon_material_a     # идеальный радиатор: ε_min = ε материала a

SIDES = {"sun": ("-", "Солнце"), "earth": ("--", "Земля")}
T_MIN_COLORS = {T: c for T, c in zip(T_MIN_LIST_C, ["tab:green", "tab:purple", "tab:orange", "tab:blue"])}

# Характерные значения размаха Δε
MARKS = {
    "конструкция": par.epsilon_eff_max - par.epsilon_eff_min,
    "предел шахматной решётки\n(gridOffset → 0)": 0.5 * (par.epsilon_material_b - par.epsilon_material_a),
    "идеал (вся площадь)": par.epsilon_material_b - par.epsilon_material_a,
}

dx = np.linspace(0.0, DX_MAX, N_DX)


def saving(S, side, T_min_c, d):
    """Экономия мощности ΔQ (Вт) при размахе Δε = d.

    Адаптивный радиатор: в холодном режиме берёт лучшее ε из [EPS_MAX - d, EPS_MAX],
    в горячем режиме достаточно, чтобы нашлось ε, проходящее по T_max.
    Фиксированный: одно ε, проходящее горячий режим, с минимальной требуемой мощностью.
    nan, если горячий режим не проходит ни при каком ε из диапазона.
    """
    eps = np.linspace(EPS_MAX - d, EPS_MAX, N_EPS)
    T_hot = tf.T_hot_epsilon(eps, S, side)
    Q_ext = tf.Q_in_cold(eps, S, side) - par.Q_cold_internal      # внешние потоки в тени
    P_rad = par.sigma * (T_min_c + 273.15) ** 4 * (S * eps + par.C)
    Q_req = np.maximum(0.0, P_rad - Q_ext)                         # требуемое внутреннее тепловыделение
    hot_ok = T_hot <= T_MAX_C + 273.15
    if not hot_ok.any():
        return float("nan")
    return float(Q_req[hot_ok].min() - Q_req.min())


def temperature_spans(S, side):
    """ΔT = T(ε_min) - T(ε_max) для горячего и холодного режимов, К."""
    eps_min = EPS_MAX - dx
    d_hot = tf.T_hot_epsilon(eps_min, S, side) - tf.T_hot_epsilon(EPS_MAX, S, side)
    d_cold = tf.T_cold_epsilon(eps_min, S, side) - tf.T_cold_epsilon(EPS_MAX, S, side)
    return d_hot, d_cold


def eps_star(S, side):
    """Наименьшее ε из [ε_a, ε_max], при котором горячий режим проходит по T_max."""
    eps = np.linspace(par.epsilon_material_a, EPS_MAX, 2000)
    ok = tf.T_hot_epsilon(eps, S, side) <= T_MAX_C + 273.15
    return float(eps[ok][0]) if ok.any() else float("nan")


# ====== Вывод в консоль ======
print(f"Допуск: T_max = {T_MAX_C:g} °C, бюджет Q_cold_internal = {par.Q_cold_internal:g} Вт")
print(f"ε_max = {EPS_MAX:.2f}; размах Δε: " +
      ", ".join(f"{name.splitlines()[0]} {val:.3f}" for name, val in MARKS.items()))
for S in S_LIST:
    print(f"\nS_рад = {S:g} м²")
    for side, (_, side_name) in SIDES.items():
        print(f"  сторона {side_name}: ε* = {eps_star(S, side):.3f} "
              f"(ниже этого ε закрытый радиатор перегревается в горячем режиме)")
        for name, d in MARKS.items():
            dq = ", ".join(f"T_min = {T:g} °C: {saving(S, side, T, d):.2f} Вт" for T in T_MIN_LIST_C)
            d_hot, d_cold = temperature_spans(S, side)
            i = int(np.argmin(np.abs(dx - d)))
            print(f"    {name.splitlines()[0]:<26} ΔQ: {dq}; "
                  f"ΔT_гор = {d_hot[i]:.1f} К, ΔT_хол = {d_cold[i]:.1f} К")

# ====== Построение ======
ncols = len(S_LIST)
q_max = 0.0
fig, axes = plt.subplots(3, ncols, figsize=(6.5 * ncols, 10), sharex=True, sharey="row", squeeze=False)

for col, S in enumerate(S_LIST):
    ax_q, ax_h, ax_c = axes[:, col]

    for side, (ls, side_name) in SIDES.items():
        for T_min_c in T_MIN_LIST_C:
            dq = np.array([saving(S, side, T_min_c, d) for d in dx])
            q_max = max(q_max, float(np.nanmax(dq)))
            ax_q.plot(dx, dq, ls, color=T_MIN_COLORS[T_min_c], linewidth=2,
                      label=f"T_min = {T_min_c:g} °C, сторона: {side_name}")
        d_hot, d_cold = temperature_spans(S, side)
        ax_h.plot(dx, d_hot, ls, color="tab:red", linewidth=2, label=f"сторона: {side_name}")
        ax_c.plot(dx, d_cold, ls, color="tab:blue", linewidth=2, label=f"сторона: {side_name}")

    title = f"S_рад = {S:g} м²"
    title += " (наш случай)" if S == par.S_rad else " (боковая грань 3U)"
    ax_q.set_title(title)
    ax_q.set_ylabel("Экономия мощности ΔQ, Вт")
    ax_h.set_ylabel("ΔT горячий режим, К")
    ax_c.set_ylabel("ΔT холодный режим, К")
    ax_c.set_xlabel(f"Размах Δε = ε_max − ε_min  (ε_max = {EPS_MAX:.2f})")
    ax_c.axhline(0, color="black", linewidth=0.8)

    for ax in (ax_q, ax_h, ax_c):
        ax.set_xlim(0, DX_MAX)
        ax.grid(True, linestyle="--", alpha=0.5)
        for val in MARKS.values():
            ax.axvline(val, color="gray", linestyle=":", linewidth=1)
    ax_q.legend(fontsize=8, loc="upper left")
    ax_h.legend(fontsize=8, loc="upper left")
    ax_c.legend(fontsize=8, loc="upper left")

axes[0, 0].set_ylim(0, max(q_max, 0.1) * 1.1)   # общий масштаб для всех столбцов

for ax in axes[0]:
    for name, val in MARKS.items():
        ax.text(val, 0.98, " " + name.replace("\n", " "), transform=ax.get_xaxis_transform(),
                rotation=90, va="top", ha="right", fontsize=7, color="gray")

fig.suptitle(f"Потенциал адаптивного радиатора в зависимости от размаха ε_eff "
             f"(T_max = {T_MAX_C:g} °C, ΔT = T(ε_закр) − T(ε_откр))")
plt.tight_layout(rect=(0, 0, 1, 0.97))
plt.show()