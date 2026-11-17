import matplotlib.pyplot as plt
import numpy as np

import thermal_model.parameters as par  # type: ignore

# ====== Параметры решётки и материалов (из parameters.py) ======
a = par.a                      # размер отверстия по X, м
gridOffset = par.gridOffset    # отступ решётки g, м

eps_a, A_a = par.epsilon_material_a, par.A_material_a   # материал a (минимальное ε)
eps_b, A_b = par.epsilon_material_b, par.A_material_b   # материал b (максимальное ε)

if hasattr(par, "epsilon_grid") and hasattr(par, "A_grid"):
    eps_g, A_g = par.epsilon_grid, par.A_grid            # материал самой решётки
else:
    eps_g, A_g = eps_b, A_b
    print("ВНИМАНИЕ: в parameters.py нет epsilon_grid / A_grid, "
          "для решётки взяты свойства материала b.\n")


# ====== Геометрия решётки (шахматное расположение отверстий) ======
def holes_fraction(a, g):
    """Доля площади радиатора, занятая отверстиями."""
    hole_height = a - 2 * g
    if hole_height <= 0:
        raise ValueError("gridOffset слишком велик: перемычки схлопнули отверстие")
    return 0.5 * hole_height / a


def fractions(delta_x, a, f_open):
    """Доли ВСЕЙ площади радиатора: (f_A, f_B, f_grid), в сумме равны 1.

    f_A, f_B - материалы a и b, видимые через отверстия (в сумме f_open),
    f_grid   - сама решётка (1 - f_open).
    """
    d = np.mod(delta_x, 2 * a)
    y = np.where(d <= a, d / a, (2 * a - d) / a)
    f_A = f_open * y
    f_B = f_open - f_A
    f_grid = np.full_like(f_A, 1.0 - f_open)
    return f_A, f_B, f_grid


f_open = holes_fraction(a, gridOffset)
dx = np.linspace(0, a, 301)                       # полный ход от "открыт" до "закрыт"
f_A, f_B, f_grid = fractions(dx, a, f_open)

eps_eff = eps_g * f_grid + eps_a * f_A + eps_b * f_B
A_eff = A_g * f_grid + A_a * f_A + A_b * f_B

eps_open, eps_closed = eps_eff[0], eps_eff[-1]    # Δx = 0 и Δx = a
A_open, A_closed = A_eff[0], A_eff[-1]
c_model = (A_closed - A_open) / (eps_closed - eps_open)

# ====== Вывод в консоль ======
print(f"a = {a * 1000:g} мм, gridOffset = {gridOffset * 1000:g} мм, "
      f"доля отверстий f_open = {f_open:.3f}")
print(f"Материал решётки: ε = {eps_g:.2f}, A = {A_g:.2f}")
print(f"Радиатор открыт (Δx = 0):  ε_eff = {eps_open:.3f}, A_eff = {A_open:.3f}")
print(f"Радиатор закрыт (Δx = a):  ε_eff = {eps_closed:.3f}, A_eff = {A_closed:.3f}")
print(f"Размах ε_eff = {abs(eps_open - eps_closed):.3f}, "
      f"наклон A(ε) = {c_model:.3f}")

# Сверка с величинами, которые использует тепловая модель
checks = [("epsilon_eff_max", eps_open), ("epsilon_eff_min", eps_closed),
          ("A_eff_at_max", A_open), ("A_eff_at_min", A_closed), ("c", c_model)]
print("\nСверка с parameters.py:")
for name, value in checks:
    if not hasattr(par, name):
        print(f"  {name}: в parameters.py нет")
    elif np.isclose(getattr(par, name), value, rtol=1e-6, atol=1e-9):
        print(f"  {name} = {getattr(par, name):.3f}: совпадает")
    else:
        print(f"  {name}: РАСХОЖДЕНИЕ, parameters.py = {getattr(par, name):.3f}, "
              f"здесь = {value:.3f}")

# ====== Построение ======
x_mm = dx * 1000
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
ax1, ax2, ax3, ax4 = axes.flat

# 1. Доли площади
ax1.plot(x_mm, f_B, label="f_B (материал b в отверстиях)", linewidth=2)
ax1.plot(x_mm, f_A, label="f_A (материал a в отверстиях)", linewidth=2)
ax1.plot(x_mm, f_grid, label="f_grid (сама решётка)", linewidth=2)
ax1.plot(x_mm, f_A + f_B + f_grid, "k--", label="сумма всех долей")
ax1.set_title("Доли площади радиатора")
ax1.set_ylabel("Доля")

# 2. Излучательность
ax2.plot(x_mm, eps_eff, linewidth=2, color="tab:blue")
ax2.set_title("Эффективная степень черноты ε_eff")
ax2.set_ylabel("ε_eff")

# 3. Поглощательность
ax3.plot(x_mm, A_eff, linewidth=2, color="tab:red")
ax3.set_title("Эффективная поглощательность A_eff")
ax3.set_ylabel("A_eff")

# 4. Связь A(ε), по которой работает тепловая модель
ax4.plot(eps_eff, A_eff, linewidth=3, color="tab:green",
         label="при сдвиге решётки")
eps_line = np.array([min(eps_open, eps_closed), max(eps_open, eps_closed)])
ax4.plot(eps_line, A_open + c_model * (eps_line - eps_open), "k--", linewidth=1,
         label="линейная модель в thermal_funcs.py")
ax4.set_title("Поглощательность в зависимости от ε_eff")
ax4.set_xlabel("ε_eff")
ax4.set_ylabel("A_eff")

# Отметки крайних состояний радиатора: (ось, y_откр, y_закр, x_откр, x_закр, сдвиг подписей)
for ax, y_open, y_closed, x_open, x_closed, off_open, off_closed in (
        (ax2, eps_open, eps_closed, 0, x_mm[-1], (8, 6), (-44, 6)),
        (ax3, A_open, A_closed, 0, x_mm[-1], (8, 6), (-44, -14)),
        (ax4, A_open, A_closed, eps_open, eps_closed, (8, 6), (8, 6))):
    ax.scatter([x_open], [y_open], color="black", zorder=5)
    ax.scatter([x_closed], [y_closed], color="black", zorder=5)
    ax.annotate("открыт", (x_open, y_open), textcoords="offset points",
                xytext=off_open, fontsize=9)
    ax.annotate("закрыт", (x_closed, y_closed), textcoords="offset points",
                xytext=off_closed, fontsize=9)

# Запас по осям, чтобы подписи и легенда не накладывались на линии
ax1.set_ylim(0, 1.35)
ax4.set_xlim(min(eps_open, eps_closed) - 0.02, max(eps_open, eps_closed) + 0.05)

for ax in (ax1, ax2, ax3):
    ax.set_xlabel("Сдвиг решётки Δx, мм")
for ax in axes.flat:
    ax.grid(True, linestyle="--", alpha=0.5)
ax1.legend(fontsize=8, loc="upper left", ncol=2)
ax4.legend(fontsize=8)

fig.suptitle(f"Радиационные свойства радиатора при сдвиге решётки: "
             f"a = {a * 1000:g} мм, gridOffset = {gridOffset * 1000:g} мм, "
             f"доля отверстий {f_open:.2f}, ε_eff от {eps_closed:.2f} до {eps_open:.2f}")
plt.tight_layout(rect=(0, 0, 1, 0.96))
plt.show()