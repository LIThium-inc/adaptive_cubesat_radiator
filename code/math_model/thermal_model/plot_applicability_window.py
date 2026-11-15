import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

import thermal_model.parameters as par  # type: ignore

# ====== Настройки ======
T_MIN_INIT_C = 0.0     # начальное значение T_min, °C
T_MAX_INIT_C = 50.0    # начальное значение T_max, °C

# Диапазон площадей радиатора, м^2
S_MIN, S_MAX = 1e-3, 5.0
N_POINTS = 800

EPS_CLOSED = par.epsilon_eff_min  # радиатор "закрыт" (минимальное достижимое ε_eff)
EPS_OPEN = par.epsilon_eff_max    # радиатор "открыт" (максимальное достижимое ε_eff)


def T_hot_S(S_rad, eps):
    """Горячий режим, температура в К (работает с массивами numpy)."""
    A_eff = par.A_eff_at_max + par.c * (eps - par.epsilon_eff_max)
    Q_in = (par.q * (S_rad * A_eff + par.C_)
            + par.Q_hot_refl + par.Q_earth + par.Q_hot_internal)
    return (Q_in / (par.sigma * (S_rad * eps + par.C))) ** 0.25


def T_cold_S(S_rad, eps):
    """Холодный режим, температура в К (Q_cold_internal уже включает нагреватель)."""
    return (par.G / (par.sigma * (S_rad * eps + par.C))) ** 0.25


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


S = np.logspace(math.log10(S_MIN), math.log10(S_MAX), N_POINTS)

# Кривые, не зависящие от температурного окна
T_hot_open = to_C(T_hot_S(S, EPS_OPEN))
T_cold_closed = to_C(T_cold_S(S, EPS_CLOSED))
T_hot_closed = to_C(T_hot_S(S, EPS_CLOSED))

# Матрицы для поиска лучшего фиксированного ε (строки: ε, столбцы: S)
eps_grid = np.linspace(EPS_CLOSED, EPS_OPEN, 100)
T_hot_grid = to_C(T_hot_S(S[None, :], eps_grid[:, None]))
T_cold_grid = to_C(T_cold_S(S[None, :], eps_grid[:, None]))


def compute_masks(T_min_c, T_max_c):
    adaptive_ok = (T_hot_open <= T_max_c) & (T_cold_closed >= T_min_c)
    fixed_ok = ((T_hot_grid <= T_max_c) & (T_cold_grid >= T_min_c)).any(axis=0)
    adaptive_needed = adaptive_ok & ~fixed_ok
    return adaptive_ok, fixed_ok, adaptive_needed


# ====== Построение ======
fig, ax = plt.subplots(figsize=(11, 7.5))
fig.subplots_adjust(bottom=0.30, top=0.93)

ax.plot(S, T_hot_open, color="tab:blue", linewidth=2,
        label=f"T_hot, ε = {EPS_OPEN} (открыт)")
ax.plot(S, T_cold_closed, color="tab:orange", linewidth=2,
        label=f"T_cold, ε = {EPS_CLOSED} (закрыт)")
ax.plot(S, T_hot_closed, color="tab:red", linewidth=1.2, linestyle="--",
        label=f"T_hot, ε = {EPS_CLOSED} (отказ в закрытом положении)")

# Для ориентира: полная поверхность 3U
ax.axvline(0.34, color="gray", linestyle=":", linewidth=1)

ax.set_xscale("log")
ax.set_xlabel("Площадь радиатора S_rad, м²")
ax.set_ylabel("Температура, °C")
ax.grid(True, which="both", linestyle="--", alpha=0.5)

# Фиксируем масштаб по Y по данным кривых, чтобы он не прыгал при движении ползунков
y_lo = min(T_cold_closed.min(), T_hot_open.min(), -20.0)
y_hi = max(T_hot_closed.max(), T_hot_open.max(), 60.0)
pad = 0.05 * (y_hi - y_lo)
ax.set_ylim(y_lo - pad, y_hi + pad)
ax.text(0.34, y_hi + pad, " вся поверхность 3U", color="gray", va="top")

# Элементы, которые перерисовываются при изменении слайдеров
dynamic = {"hline_max": None, "hline_min": None, "band": None,
           "ok": None, "needed": None}

info_text = fig.text(0.5, 0.20, "", ha="center", va="center", fontsize=9)

# ====== Слайдеры ======
ax_tmin = fig.add_axes([0.15, 0.10, 0.70, 0.03])
ax_tmax = fig.add_axes([0.15, 0.05, 0.70, 0.03])
slider_tmin = Slider(ax_tmin, "T_min, °C", -60.0, 20.0, valinit=T_MIN_INIT_C, valstep=1.0)
slider_tmax = Slider(ax_tmax, "T_max, °C", 20.0, 120.0, valinit=T_MAX_INIT_C, valstep=1.0)


def update(_=None):
    T_min_c = slider_tmin.val
    T_max_c = slider_tmax.val
    adaptive_ok, fixed_ok, adaptive_needed = compute_masks(T_min_c, T_max_c)

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
        color="tab:green", alpha=0.18,
        label="Адаптивный радиатор удовлетворяет допуску")
    dynamic["needed"] = ax.fill_between(
        S, 0, 1, where=adaptive_needed, transform=ax.get_xaxis_transform(),
        facecolor="tab:green", alpha=0.25, hatch="//", edgecolor="green",
        label="... и фиксированное ε не удовлетворяет")

    ax.set_title(f"Область применимости адаптивного радиатора: "
                 f"допуск [{T_min_c:g}, {T_max_c:g}] °C")
    ax.legend(loc="upper left", fontsize=9)

    info_text.set_text(
        f"Адаптивный радиатор подходит: {format_intervals(adaptive_ok, S)}\n"
        f"Фиксированное ε подходит: {format_intervals(fixed_ok, S)}\n"
        f"Адаптивность даёт выигрыш: {format_intervals(adaptive_needed, S)}"
    )

    fig.canvas.draw_idle()


slider_tmin.on_changed(update)
slider_tmax.on_changed(update)
update()

plt.show()