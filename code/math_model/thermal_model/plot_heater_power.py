import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

import thermal_model.parameters as par  # type: ignore

# ====== Настройки ======
T_MIN_INIT_C = 0.0     # начальное значение T_min, °C
T_MAX_INIT_C = 50.0    # начальное значение T_max, °C

S_MIN, S_MAX = 1e-4, 0.05   # диапазон площади радиатора, м^2
N_POINTS = 500

# Характерные площади для 3U: торец 10x10 см и боковая грань 10x30 см
S_MARKS = {0.01: "торец 3U (0.01 м²)", 0.03: "боковая грань 3U (0.03 м²)"}

EPS_CLOSED = par.epsilon_eff_min  # радиатор "закрыт" (минимальное достижимое ε_eff)
EPS_OPEN = par.epsilon_eff_max    # радиатор "открыт" (максимальное достижимое ε_eff)

Q_BUDGET = par.Q_cold_internal       # бюджет внутреннего тепловыделения (электроника + нагреватель)


def T_hot_S(S_rad, eps):
    """Горячий режим, температура в К (работает с массивами numpy)."""
    A_eff = par.A_eff_at_max + par.c * (eps - par.epsilon_eff_max)
    Q_in = (par.q * (S_rad * A_eff + par.C_)
            + par.Q_hot_refl + par.Q_earth + par.Q_hot_internal)
    return (Q_in / (par.sigma * (S_rad * eps + par.C))) ** 0.25


def q_required(S_rad, eps, T_min_K):
    """Требуемое суммарное внутреннее тепловыделение (Вт) в холодном режиме."""
    P_rad = par.sigma * T_min_K ** 4 * (S_rad * eps + par.C)
    return np.maximum(0.0, P_rad - par.Q_earth)


def eps_required_hot(S_rad, T_max_K):
    """Минимальное ε, при котором T_hot <= T_max (T_hot убывает с ε).

    nan там, где горячий режим не проходит даже при ε = EPS_OPEN.
    """
    S_rad = np.asarray(S_rad, dtype=float)
    open_ok = T_hot_S(S_rad, EPS_OPEN) <= T_max_K
    closed_ok = T_hot_S(S_rad, EPS_CLOSED) <= T_max_K

    lo = np.full_like(S_rad, EPS_CLOSED)
    hi = np.full_like(S_rad, EPS_OPEN)
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        too_hot = T_hot_S(S_rad, mid) > T_max_K
        lo = np.where(too_hot, mid, lo)
        hi = np.where(too_hot, hi, mid)

    eps = hi
    eps = np.where(closed_ok, EPS_CLOSED, eps)
    eps = np.where(open_ok, eps, np.nan)
    return eps


def compute(T_min_c, T_max_c, S_arr):
    """(Q_adaptive, Q_fixed, hot_ok_mask) для заданного температурного окна."""
    T_min_K = T_min_c + 273.15
    T_max_K = T_max_c + 273.15
    eps_fix = eps_required_hot(S_arr, T_max_K)
    hot_ok = ~np.isnan(eps_fix)
    q_fix = q_required(S_arr, eps_fix, T_min_K)
    q_adapt = np.where(hot_ok, q_required(S_arr, EPS_CLOSED, T_min_K), np.nan)
    return q_adapt, q_fix, hot_ok


S = np.linspace(S_MIN, S_MAX, N_POINTS)

# ====== Построение ======
fig, ax = plt.subplots(figsize=(11, 7))
fig.subplots_adjust(bottom=0.30, top=0.92)

q_adapt, q_fix, hot_ok = compute(T_MIN_INIT_C, T_MAX_INIT_C, S)

(line_fix,) = ax.plot(S, q_fix, color="tab:gray", linewidth=2, label="Фиксированное ε")
(line_adapt,) = ax.plot(S, q_adapt, color="tab:green", linewidth=2,
                        label="Адаптивный радиатор")
ax.axhline(Q_BUDGET, color="tab:red", linestyle="--", linewidth=1.5,
           label=f"Бюджет Q_cold_internal = {Q_BUDGET:g} Вт")
for s_mark in S_MARKS:
    ax.axvline(s_mark, color="black", linestyle=":", linewidth=1)

# Элементы, которые перерисовываются при изменении слайдеров
dynamic = {"saving": None, "infeasible": None}

ax.set_xlim(0, S_MAX)
ax.set_ylim(bottom=0)
ax.set_xlabel("Площадь радиатора S_rad, м²")
ax.set_ylabel("Требуемая мощность, Вт")
ax.grid(True, linestyle="--", alpha=0.5)

info_text = fig.text(0.5, 0.20, "", ha="center", va="center", fontsize=9)

# ====== Слайдеры ======
ax_tmin = fig.add_axes([0.15, 0.10, 0.70, 0.03])
ax_tmax = fig.add_axes([0.15, 0.05, 0.70, 0.03])
slider_tmin = Slider(ax_tmin, "T_min, °C", -60.0, 20.0, valinit=T_MIN_INIT_C, valstep=1.0)
slider_tmax = Slider(ax_tmax, "T_max, °C", 20.0, 120.0, valinit=T_MAX_INIT_C, valstep=1.0)


def update(_=None):
    T_min_c = slider_tmin.val
    T_max_c = slider_tmax.val
    q_a, q_f, ok = compute(T_min_c, T_max_c, S)

    line_adapt.set_ydata(q_a)
    line_fix.set_ydata(q_f)

    # Перерисовываем заливки
    for key in dynamic:
        if dynamic[key] is not None:
            dynamic[key].remove()
            dynamic[key] = None

    dynamic["saving"] = ax.fill_between(S, q_a, q_f, where=ok, color="tab:green",
                                        alpha=0.2, label="Экономия мощности")
    if (~ok).any():
        dynamic["infeasible"] = ax.fill_between(
            S, 0, 1, where=~ok, transform=ax.get_xaxis_transform(),
            color="tab:red", alpha=0.10,
            label=f"T_hot > {T_max_c:g} °C даже при открытом радиаторе")

    # Масштаб по Y
    finite = np.concatenate([q_a[np.isfinite(q_a)], q_f[np.isfinite(q_f)]])
    top = max(Q_BUDGET, finite.max() if finite.size else 0.0) * 1.15
    ax.set_ylim(0, max(top, 1.0))

    ax.set_title(f"Суммарное внутреннее тепловыделение в холодном режиме: "
                 f"T_min = {T_min_c:g} °C, T_max = {T_max_c:g} °C")
    ax.legend(loc="upper left", fontsize=9)

    # Текст с итогами
    lines = []
    if ok.all():
        lines.append("Горячий режим проходит на всём диапазоне площадей")
    elif ok.any():
        lines.append(f"Горячий режим проходит при S_rad ≥ {S[ok][0]:.4f} м²")
    else:
        lines.append("Горячий режим не проходит ни при какой площади из диапазона")

    for s_mark, label in S_MARKS.items():
        qa, qf, ok_mark = compute(T_min_c, T_max_c, np.array([s_mark]))
        if not ok_mark[0]:
            lines.append(f"{label}: горячий режим не проходит")
        else:
            lines.append(f"{label}: адаптивный {qa[0]:.1f} Вт, фиксированный {qf[0]:.1f} Вт, "
                         f"экономия {qf[0] - qa[0]:.1f} Вт")
    info_text.set_text("\n".join(lines))

    fig.canvas.draw_idle()


slider_tmin.on_changed(update)
slider_tmax.on_changed(update)
update()

plt.show()