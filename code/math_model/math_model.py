import matplotlib.pyplot as plt


def calculate_holes_fraction(xDim, yDim, gridStep):
    return (xDim * yDim) / (gridStep * gridStep)


def material_fraction(deltaX, gridStep, gridOffset, holesFraction):
    holeSize = gridStep - gridOffset
    return (holesFraction * (1 - (deltaX % holeSize) / holeSize), holesFraction * ((deltaX % holeSize) / holeSize))
