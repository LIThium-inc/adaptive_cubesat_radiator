import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, CubicSpline


"""
# 1. Читаем данные
df = pd.read_csv("data.csv")
df = df.sort_values("x").drop_duplicates("x")   # x должен быть отсортирован и без повторов
x = df["x"].to_numpy()
y = df["y"].to_numpy()

# 2. Строим интерполяторы
f_linear = interp1d(x, y, kind="linear")
f_cubic = CubicSpline(x, y)

# 3. Считаем значения в новых точках
x_new = np.linspace(x.min(), x.max(), 200)
y_linear = f_linear(x_new)
y_cubic = f_cubic(x_new)
"""



class myFunc():
    def __init__(self, dataFileName: str):
        self.df = pd.read_csv(dataFileName)
        self.df = self.df.sort_values("x").drop_duplicates("x")
        self.x = self.df["x"].to_numpy()
        self.y = self.df["y"].to_numpy()
        self.f_linear = interp1d(self.x, self.y, kind="linear")
        self.f_cubic = CubicSpline(self.x, self.y)


    def get_value(self, x_new: float, method: str = "cubic") -> float:
        if method == "linear":
            return self.f_linear(x_new)
        elif method == "cubic":
            return self.f_cubic(x_new)
        else:
            raise ValueError("Метод должен быть 'linear' или 'cubic'")