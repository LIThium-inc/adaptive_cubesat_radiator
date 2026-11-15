import numpy as np
import matplotlib.pyplot as plt

from .parameters import epsilon_eff_max, epsilon_eff_min, A_eff_at_max, A_eff_at_min
from .thermal_funcs import T_hot_epsilon, T_cold_epsilon

x_min, x_max = min(epsilon_eff_min, epsilon_eff_max), max(epsilon_eff_min, epsilon_eff_max)
print("Средняя эффективная степень черноты: min =", x_min, ", max =", x_max)
# Создаём массив точек (чем больше точек, тем плавнее кривая)
x = np.linspace(x_min, x_max, 500)
y1 = [T_hot_epsilon(epsilon) for epsilon in x]  # Вычисляем значения функции для каждой точки x
y2 = [T_cold_epsilon(epsilon) for epsilon in x]  # Вычисляем значения функции для каждой точки x

# Строим график
plt.figure(figsize=(8, 5))
plt.plot(x, y1, label='T_hot(epsilon)', linewidth=2)
plt.plot(x, y2, label='T_cold(epsilon)', linewidth=2)

# Подписи и легенда
plt.xlabel('Средняя эффективная степень черноты')
plt.ylabel('Температура (K)')
plt.title('График функций T горячего и T холодного режимов в зависимости от средней эффективной степени черноты')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

# Устанавливаем пределы по осям (чтобы график был ровно в нужных границах)
plt.xlim(x_min, x_max)
# Опционально: можно задать и ylim, если хочешь фиксированный масштаб по Y
# plt.ylim(-10, 50)

plt.tight_layout()
plt.show()