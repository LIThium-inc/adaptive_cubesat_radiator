import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import RadioButtons, Slider

import thermal_model.parameters as par  # type: ignore
import thermal_model.thermal_funcs as tf  # type: ignore

# ====== Настройки ======
T_MIN_INIT_C = 0.0     # начальное значение T_min, °C
T_MAX_INIT_C = 50.0    # начальное значение T_max, °C
SIDE_INIT = getattr(par, "RADIATOR_SIDE", "sun")   # "sun" или "earth"

S_MIN, S_MAX = 1e-4, 0.05   # диапазон площади радиатора, м^2
N_POINTS = 500
N_EPS = 150                 # число шагов по ε внутри диапазона [ε_min, ε_max]

# Характерные площади одной грани 3U: торец 10x10 см и боковая грань 10x30 см
S_MARKS = {0.01: "торец", 0.03: "боковая грань"}

SIDE_LABELS = {"sun": "Солнце", "earth": "Земля"}

EPS_MIN = par.epsilon_eff_min   # радиатор "закрыт"
EPS_MAX = par.epsilon_eff_max   # радиатор "открыт"

# Бюджет суммарного внутреннего тепловыделения в холодном режиме (электроника + нагреватель)
Q_BUDGET = par.Q_cold_internal

S = np.linspace(S_MIN, S_MAX, N_POINTS)
EPS = np.linspace(EPS_MIN, EPS_MAX, N_EPS)[:, None]   # строки: ε, столбцы: S


def intervals(mask, x):
    """Список (x_start, x_end) для непрерывных участков, где mask == True."""
    result = []
    start = None
    for i, ok in enumerate(mask):
        if ok and start is None:
            start = x[i]
        if not ok and start is not None:
            result.append((start, x[i - 1]))
            start = None
    if start is not None:
        result.append((start, x[-1]))
    return result


def compute(side, T_min_c, T_max_c, S_arr):
    """Требуемое суммарное внутреннее тепловыделение в холодном режиме, Вт.

    Баланс в холодном режиме: σ·T_min⁴·(S·ε + C) = Q_внешнее + Q_внутр,
    где Q_внешнее берётся из thermal_funcs (без Q_cold_internal).
    Возвращает (q_adaptive, q_fixed, hot_feasible):
      - адаптивный: в горячем режиме подбирается ε, проходящее по T_max,
        в холодном - ε с минимальной требуемой мощностью из всего диапазона;
      - фиксированный: одно ε, которое проходит горячий режим, из них берётся
        то, что даёт минимальную требуемую мощность.
    """
    S_row = np.asarray(S_arr, dtype=float)[None, :]
    T_hot = tf.T_hot_epsilon(EPS, S_row, side)                       # К
    Q_ext = tf.Q_in_cold(EPS, S_row, side) - par.Q_cold_internal     # внешние потоки в тени
    P_rad = par.sigma * (T_min_c + 273.15) ** 4 * (S_row * EPS + par.C)
    Q_req = np.maximum(0.0, P_rad - Q_ext)

    hot_ok = T_hot <= T_max_c + 273.15
    feasible = hot_ok.any(axis=0)
    q_adapt = np.where(feasible, Q_req.min(axis=0), np.nan)
    q_fix = np.where(feasible, np.where(hot_ok, Q_req, np.inf).min(axis=0), np.nan)
    return q_adapt, q_fix, feasible


# ====== Построение ======
fig, ax = plt.subplots(figsize=(12, 8))
fig.subplots_adjust(bottom=0.32, top=0.93)

q_a0, q_f0, _ = compute(SIDE_INIT, T_MIN_INIT_C, T_MAX_INIT_C, S)
(line_fix,) = ax.plot(S, q_f0, color="tab:gray", linewidth=2, label="Фиксированное ε")
(line_adapt,) = ax.plot(S, q_a0, color="tab:green", linewidth=2, label="Адаптивный радиатор")
ax.axhline(Q_BUDGET, color="tab:red", linestyle="--", linewidth=1.5,
           label=f"Бюджет Q_cold_internal = {Q_BUDGET:g} Вт")
for s_mark, label in S_MARKS.items():
    ax.axvline(s_mark, color="black", linestyle=":", linewidth=1)
    ax.text(s_mark, 0.98, " " + label, transform=ax.get_xaxis_transform(),
            rotation=90, va="top", ha="right", fontsize=8)

ax.set_xlim(0, S_MAX)
ax.set_xlabel("Площадь радиатора S_rad, м²")
ax.set_ylabel("Требуемая мощность, Вт")
ax.grid(True, linestyle="--", alpha=0.5)

dynamic = {"saving": None, "infeasible": None}
state = {"side": SIDE_INIT}

info_text = fig.text(0.64, 0.205, "", ha="center", va="center", fontsize=9)

# ====== Элементы управления ======
ax_radio = fig.add_axes([0.03, 0.03, 0.17, 0.15])
ax_radio.set_title("Сторона радиатора", fontsize=9)
radio = RadioButtons(ax_radio, list(SIDE_LABELS.values()),
                     active=list(SIDE_LABELS).index(SIDE_INIT))

ax_tmin = fig.add_axes([0.35, 0.10, 0.55, 0.03])
ax_tmax = fig.add_axes([0.35, 0.05, 0.55, 0.03])
slider_tmin = Slider(ax_tmin, "T_min, °C", -60.0, 20.0, valinit=T_MIN_INIT_C, valstep=1.0)
slider_tmax = Slider(ax_tmax, "T_max, °C", 20.0, 120.0, valinit=T_MAX_INIT_C, valstep=1.0)


def update(_=None):
    side = state["side"]
    T_min_c, T_max_c = slider_tmin.val, slider_tmax.val
    q_a, q_f, ok = compute(side, T_min_c, T_max_c, S)

    line_adapt.set_ydata(q_a)
    line_fix.set_ydata(q_f)

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
            label=f"T_hot > {T_max_c:g} °C при любом ε из диапазона")

    finite = np.concatenate([q_a[np.isfinite(q_a)], q_f[np.isfinite(q_f)]])
    top = max(Q_BUDGET, finite.max() if finite.size else 0.0) * 1.15
    ax.set_ylim(0, max(top, 1.0))

    ax.set_title(f"Суммарное внутреннее тепловыделение в холодном режиме, "
                 f"сторона: {SIDE_LABELS[side]}, T_min = {T_min_c:g} °C, T_max = {T_max_c:g} °C")
    ax.legend(loc="center right", fontsize=8)

    lines = []
    if ok.all():
        lines.append("Горячий режим проходит на всём диапазоне площадей")
    elif ok.any():
        ivs = "; ".join(f"{lo:.4f}…{hi:.4f} м²" for lo, hi in intervals(ok, S))
        lines.append(f"Горячий режим проходит при S_rad: {ivs}")
    else:
        lines.append("Горячий режим не проходит ни при какой площади из диапазона")

    marks = np.array(list(S_MARKS))
    qa_m, qf_m, ok_m = compute(side, T_min_c, T_max_c, marks)
    for s_mark, label, qa, qf, okm in zip(marks, S_MARKS.values(), qa_m, qf_m, ok_m):
        if not okm:
            lines.append(f"{label} ({s_mark:g} м²): горячий режим не проходит")
        else:
            lines.append(f"{label} ({s_mark:g} м²): адаптивный {qa:.1f} Вт, "
                         f"фиксированный {qf:.1f} Вт, экономия {qf - qa:.2f} Вт")
    info_text.set_text("\n".join(lines))
    fig.canvas.draw_idle()


def on_side(label):
    state["side"] = next(k for k, v in SIDE_LABELS.items() if v == label)
    update()


radio.on_clicked(on_side)
slider_tmin.on_changed(update)
slider_tmax.on_changed(update)
update()

plt.show()