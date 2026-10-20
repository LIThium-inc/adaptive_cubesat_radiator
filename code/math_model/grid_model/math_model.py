def calculate_holes_fraction(a: float, gridOffset: float, checkerboard: bool = True) -> float:
    holeHeight = a - gridOffset * 2
    if holeHeight <= 0:
        raise ValueError("gridOffset слишком велик: перемычки схлопнули отверстие по высоте")
    fraction = holeHeight / a 
    if checkerboard:
        fraction *= 0.5
    else:
        raise ValueError("не реализован вариант без шахматного порядка")
    return fraction


def material_fraction(deltaX: float, a: float, f_open: float) -> tuple[float, float]:
    d = deltaX % (2 * a)
    y = d / a if d <= a else (2 * a - d) / a
    f_A = f_open * y
    f_B = f_open - f_A
    return f_B, f_A