import matplotlib.pyplot as plt
import numpy as np

import thermal_model.parameters as par  # type: ignore

# ====== Параметры геометрии (для проверки можно задать свои) ======
a = par.a                      # размер отверстия по X, м
gridOffset = par.gridOffset    # отступ g, м
# a = 0.01
# gridOffset = 0.001

# ====== Свойства материалов ======
eps_a, A_a = par.epsilon_material_a, par.A_material_a   # материал a
eps_b, A_b = par.epsilon_material_b, par.A_material_b   # материал b

# Материал самой решётки (пластины с отверстиями). ПОДСТАВЬТЕ СВОЙ.
# "a" - решётка из материала a, "b" - из материала b, "custom" - свои значения ниже.
GRID_MATERIAL = "b"
GRID_EPS_CUSTOM = 0.90
GRID_A_CUSTOM = 0.27

if GRID_MATERIAL == "a":
    eps_g, A_g = eps_a, A_a
elif GRID_MATERIAL == "b":
    eps_g, A_g = eps_b, A_b
else:
    eps_g, A_g = GRID_EPS_CUSTOM, GRID_A_CUSTOM


# ====== Геометрия (та же логика, что в вашем math_model.py) ======
def holes_fraction(a, g):
    """Доля площади, занятая отверстиями (шахматный порядок)."""
    hole_height = a - 2 * g
    if hole_height <= 0:
        raise ValueError("gridOffset слишком велик")
    return 0.5 * hole_height / a


def fractions(delta_x, a, f_open):
    """Возвращает (f_A, f_B, f_grid) - доли ВСЕЙ площади радиатора.

    f_A, f_B - материалы a и b, видимые через отверстия (в сумме f_open),
    f_grid   - сама решётка (1 - f_open). Сумма всех трёх равна 1.
    """
    d = np.mod(delta_x, 2 * a)
    y = np.where(d <= a, d / a, (2 * a - d) / a)
    f_A = f_open * y
    f_B = f_open - f_A
    f_grid = np.full_like(f_A, 1.0 - f_open)
    return f_A, f_B, f_grid


f_open = holes_fraction(a, gridOffset)
dx = np.linspace(0, a, 301)
f_A, f_B, f_grid = fractions(dx, a, f_open)

# Как в текущем коде: решётка не учитывается, f_A + f_B = f_open
eps_naive = eps_a * f_A + eps_b * f_B
A_naive = A_a * f_A + A_b * f_B

# С учётом решётки: суммируем по всей площади
eps_eff = eps_g * f_grid + eps_a * f_A + eps_b * f_B
A_eff = A_g * f_grid + A_a * f_A + A_b * f_B

# Наклон линии A(ε): сейчас через точки a и b, реально - через концы диапазона
c_old = (A_a - A_b) / (eps_a - eps_b)
c_eff = (A_eff[-1] - A_eff[0]) / (eps_eff[-1] - eps_eff[0])

print(f"a = {a * 1000:g} мм, gridOffset = {gridOffset * 1000:g} мм, "
      f"доля отверстий f_open = {f_open:.3f}")
print(f"Материал решётки: {GRID_MATERIAL} (ε = {eps_g}, A = {A_g})")
print(f"Δx = 0: ε_eff = {eps_eff[0]:.3f}, A_eff = {A_eff[0]:.3f}")
print(f"Δx = a: ε_eff = {eps_eff[-1]:.3f}, A_eff = {A_eff[-1]:.3f}")
print(f"Размах ε_eff = {abs(eps_eff[0] - eps_eff[-1]):.3f} "
      f"(максимум возможный при такой f_open: {(eps_b - eps_a) * f_open:.3f})")
print(f"Наклон A(ε): по точкам a и b c = {c_old:.3f}, по реальным концам c_eff = {c_eff:.3f}")
print(f"Как сейчас в коде (без решётки): ε(Δx=0) = {eps_naive[0]:.3f}, "
      f"ε(Δx=a) = {eps_naive[-1]:.3f}")

# ====== Построение ======
x_mm = dx * 1000
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
ax1, ax2, ax3, ax4 = axes.flat

# 1. Доли площади
ax1.plot(x_mm, f_A, label="f_A (материал a в отверстиях)", linewidth=2)
ax1.plot(x_mm, f_B, label="f_B (материал b в отверстиях)", linewidth=2)
ax1.plot(x_mm, f_grid, label="f_grid (сама решётка)", linewidth=2)
ax1.plot(x_mm, f_A + f_B + f_grid, "k--", label="сумма всех долей")
ax1.plot(x_mm, f_A + f_B, ":", color="gray", label="f_A + f_B (как сейчас в коде)")
ax1.set_title("Доли площади радиатора")
ax1.set_ylabel("Доля")

# 2. Излучательность
ax2.plot(x_mm, eps_eff, linewidth=2, label="ε_eff с учётом решётки")
ax2.plot(x_mm, eps_naive, "--", color="gray", label="без решётки (как сейчас)")
ax2.set_title("Эффективная степень черноты ε_eff")
ax2.set_ylabel("ε_eff")

# 3. Поглощательность
ax3.plot(x_mm, A_eff, linewidth=2, color="tab:red", label="A_eff с учётом решётки")
ax3.plot(x_mm, A_naive, "--", color="gray", label="без решётки (как сейчас)")
ax3.set_title("Эффективная поглощательность A_eff")
ax3.set_ylabel("A_eff")

# 4. Связь A(ε)
eps_line = np.linspace(eps_a, eps_b, 50)
ax4.plot(eps_line, A_b + c_old * (eps_line - eps_b), "--", color="gray",
         label="линия через точки a и b (используется в T_hot)")
ax4.plot(eps_eff, A_eff, linewidth=3, color="tab:green",
         label="реальная зависимость при сдвиге решётки")
ax4.scatter([eps_a, eps_b], [A_a, A_b], color="black", zorder=5)
ax4.annotate("a", (eps_a, A_a), textcoords="offset points", xytext=(6, 6))
ax4.annotate("b", (eps_b, A_b), textcoords="offset points", xytext=(6, 6))
ax4.set_title("Поглощательность в зависимости от ε")
ax4.set_xlabel("ε_eff")
ax4.set_ylabel("A_eff")

for ax in (ax1, ax2, ax3):
    ax.set_xlabel("Сдвиг решётки Δx, мм")
for ax in axes.flat:
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=8)

fig.suptitle(f"Проверка модели решётки: a = {a * 1000:g} мм, "
             f"gridOffset = {gridOffset * 1000:g} мм, f_open = {f_open:.2f}, "
             f"решётка из материала {GRID_MATERIAL}")
plt.tight_layout(rect=(0, 0, 1, 0.97))
plt.show()