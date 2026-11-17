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

S_MIN, S_MAX = 1e-3, 0.5    # диапазон площадей радиатора, м^2
N_POINTS = 600
N_EPS = 150                 # число шагов по ε внутри диапазона [ε_min, ε_max]

# Характерные площади одной грани 3U: торец 10x10 см и боковая грань 10x30 см
S_MARKS = {0.01: "торец", 0.03: "боковая грань"}

SIDE_LABELS = {"sun": "Солнце", "earth": "Земля"}

EPS_MIN = par.epsilon_eff_min   # радиатор "закрыт"
EPS_MAX = par.epsilon_eff_max   # радиатор "открыт"

S = np.logspace(math.log10(S_MIN), math.log10(S_MAX), N_POINTS)
EPS = np.linspace(EPS_MIN, EPS_MAX, N_EPS)[:, None]   # строки: ε, столбцы: S


def to_C(T_kelvin):
    return T_kelvin - 273.15


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


def format_intervals(mask, x):
    ivs = intervals(mask, x)
    if not ivs:
        return "нет"
    return "; ".join(f"{lo:.3g}…{hi:.3g} м²" for lo, hi in ivs)


# Температурные поля для каждой стороны считаются один раз:
# строка - значение ε из диапазона, столбец - площадь радиатора.
FIELDS = {}
for _side in SIDE_LABELS:
    T_hot = to_C(tf.T_hot_epsilon(EPS, S[None, :], _side))
    T_cold = to_C(tf.T_cold_epsilon(EPS, S[None, :], _side))
    cols = np.arange(S.size)
    FIELDS[_side] = {
        "T_hot": T_hot,
        "T_cold": T_cold,
        # Лучшее ε для каждого режима выбирается независимо (адаптивный радиатор)
        "T_hot_best": T_hot.min(axis=0),
        "T_cold_best": T_cold.max(axis=0),
        # Отказ: радиатор остался в положении, оптимальном для холодного режима
        "T_hot_fail": T_hot[T_cold.argmax(axis=0), cols],
    }


def compute_masks(side, T_min_c, T_max_c):
    f = FIELDS[side]
    hot_ok = f["T_hot"] <= T_max_c
    cold_ok = f["T_cold"] >= T_min_c
    adaptive_ok = hot_ok.any(axis=0) & cold_ok.any(axis=0)   # ε можно менять между режимами
    fixed_ok = (hot_ok & cold_ok).any(axis=0)                # одно ε на оба режима
    return adaptive_ok, fixed_ok, adaptive_ok & ~fixed_ok


# ====== Построение ======
fig, ax = plt.subplots(figsize=(12, 8))
fig.subplots_adjust(bottom=0.32, top=0.93)

(line_hot,) = ax.plot(S, FIELDS[SIDE_INIT]["T_hot_best"], color="tab:blue", linewidth=2)
(line_cold,) = ax.plot(S, FIELDS[SIDE_INIT]["T_cold_best"], color="tab:orange", linewidth=2)
(line_fail,) = ax.plot(S, FIELDS[SIDE_INIT]["T_hot_fail"], color="tab:red", linewidth=1.2,
                       linestyle="--")

ax.set_xscale("log")
ax.set_xlabel("Площадь радиатора S_rad, м²")
ax.set_ylabel("Температура, °C")
ax.grid(True, which="both", linestyle="--", alpha=0.5)
for s_mark, label in S_MARKS.items():
    ax.axvline(s_mark, color="gray", linestyle=":", linewidth=1)
    ax.text(s_mark, 0.02, " " + label, transform=ax.get_xaxis_transform(),
            rotation=90, va="bottom", ha="right", fontsize=8, color="gray")

# Масштаб по Y общий для обеих сторон, чтобы он не прыгал
all_lo = min(min(f["T_cold_best"].min(), f["T_hot_best"].min()) for f in FIELDS.values())
all_hi = max(max(f["T_hot_fail"].max(), f["T_hot_best"].max()) for f in FIELDS.values())
y_lo, y_hi = min(all_lo, -20.0), max(all_hi, 60.0)
pad = 0.05 * (y_hi - y_lo)
ax.set_ylim(y_lo - pad, y_hi + pad)

dynamic = {"hline_max": None, "hline_min": None, "band": None, "ok": None, "needed": None}
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
    f = FIELDS[side]
    T_min_c, T_max_c = slider_tmin.val, slider_tmax.val
    adaptive_ok, fixed_ok, adaptive_needed = compute_masks(side, T_min_c, T_max_c)

    line_hot.set_ydata(f["T_hot_best"])
    line_cold.set_ydata(f["T_cold_best"])
    line_fail.set_ydata(f["T_hot_fail"])
    line_hot.set_label(f"T_hot, лучшее ε ∈ [{EPS_MIN:.2f}, {EPS_MAX:.2f}]")
    line_cold.set_label(f"T_cold, лучшее ε ∈ [{EPS_MIN:.2f}, {EPS_MAX:.2f}]")
    line_fail.set_label("T_hot при отказе (радиатор в положении, оптимальном для холода)")

    for key, artist in dynamic.items():
        if artist is not None:
            artist.remove()
            dynamic[key] = None

    dynamic["hline_max"] = ax.axhline(T_max_c, color="black", linestyle=":", linewidth=1)
    dynamic["hline_min"] = ax.axhline(T_min_c, color="black", linestyle=":", linewidth=1)
    dynamic["band"] = ax.axhspan(T_min_c, T_max_c, color="gray", alpha=0.08,
                                 label=f"Допуск объекта [{T_min_c:g}, {T_max_c:g}] °C")
    dynamic["ok"] = ax.fill_between(
        S, 0, 1, where=adaptive_ok, transform=ax.get_xaxis_transform(),
        color="tab:green", alpha=0.18, label="Адаптивный радиатор удовлетворяет допуску")
    dynamic["needed"] = ax.fill_between(
        S, 0, 1, where=adaptive_needed, transform=ax.get_xaxis_transform(),
        facecolor="tab:green", alpha=0.25, hatch="//", edgecolor="green",
        label="... и фиксированное ε не удовлетворяет")

    ax.set_title(f"Область применимости адаптивного радиатора, "
                 f"сторона: {SIDE_LABELS[side]}, допуск [{T_min_c:g}, {T_max_c:g}] °C")
    ax.legend(loc="upper left", fontsize=8)

    info_text.set_text(
        f"Адаптивный радиатор подходит: {format_intervals(adaptive_ok, S)}\n"
        f"Фиксированное ε подходит: {format_intervals(fixed_ok, S)}\n"
        f"Адаптивность даёт выигрыш: {format_intervals(adaptive_needed, S)}")
    fig.canvas.draw_idle()


def on_side(label):
    state["side"] = next(k for k, v in SIDE_LABELS.items() if v == label)
    update()


radio.on_clicked(on_side)
slider_tmin.on_changed(update)
slider_tmax.on_changed(update)
update()

plt.show()