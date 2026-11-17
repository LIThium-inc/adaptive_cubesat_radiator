from thermal_model.grid_model import math_model as grid_math_model #type: ignore

#Constants for thermal calculations
sigma = 5.67 * 10 ** -8 #Stefan-Boltzmann constant in W/m^2K^4
q = 1361 #Solar constant in W/m^2
R_earth = 6371 * 10 ** 3 #Radius of earth in m
Albedo_earth = 0.3 #Albedo of earth

#Parameters for Geoscan 3U v.2 CubeSat platform
S_rad = 0.008 #Area of radiator in m^2
S_solar = 0.08 #Area of solar panel in m^2
S_solar_earth = 0.024 #Area of solar panel facing earth in m^2
S_solar_sun = 0.016 #Area of solar panel facing sun in m^2
S_end_pannels = 0.02 #Area of end pannels in m^2

A_solar = 0.92 #Abbsorptivity of solar panel
epsilon_solar = 0.82 #Emissivity of solar panel

epsilon_end_pannels = 0.85 #Emissivity of end pannels

A_material_a = 0.94 #Abbsorptivity of material a
epsilon_material_a = 0.15 #Emissivity of material a

A_material_b = 0.27 #Abbsorptivity of material b
epsilon_material_b = 0.90 #Emissivity of material b

A_grid = A_material_b #Abbsorptivity of grid material
epsilon_grid = epsilon_material_b #Emissivity of grid material

gridOffset = 0.001 #Offset of grid in m
a = 0.01 #Length of hole in grid in m

epsilon_eff_max, A_eff_at_max = grid_math_model.effective_properties(
    0, a, gridOffset, epsilon_material_a, A_material_a,
    epsilon_material_b, A_material_b, epsilon_grid, A_grid)

epsilon_eff_min, A_eff_at_min = grid_math_model.effective_properties(
    a, a, gridOffset, epsilon_material_a, A_material_a,
    epsilon_material_b, A_material_b, epsilon_grid, A_grid)

c = (A_eff_at_min - A_eff_at_max) / (epsilon_eff_min - epsilon_eff_max)

Q_hot_internal = 10 #Internal heat generation in W for hot case
Q_cold_internal = 7 #Internal heat generation in W for cold case

h = 500 * 10 ** 3 #Altitude of satellite in m

phi = (R_earth / (R_earth + h))**2 #View factor of earth from satellite

Q_hot_refl = S_solar_earth * A_solar * q * Albedo_earth * phi #Reflected solar radiation from earth in W

Q_earth = S_solar_earth * epsilon_solar * phi * ((1 - Albedo_earth) * q) / 4 #Earth radiation in W

K_ins = 1.0 #Effective radiating fraction of constant surfaces (1 = no insulation, <1 = MLI / low-emissivity coating)

C = K_ins * (S_solar * epsilon_solar + S_end_pannels * epsilon_end_pannels)

C_ = S_solar_sun * A_solar

G = Q_earth + Q_cold_internal

RADIATOR_SIDE = "sun"   # "sun" - радиатор на солнечной стороне, "earth" - на стороне к Земле

S_panel_one = 0.008   # Area of one solar panel (one face) in m^2

# Число панелей, обращённых к Солнцу / к Земле, в зависимости от стороны радиатора
N_PANELS_SUN   = {"sun": 2, "earth": 3}
N_PANELS_EARTH = {"sun": 3, "earth": 2}

# Удельные потоки от Земли на поверхность, обращённую к Земле, Вт/м^2
q_earth_ir = phi * (1 - Albedo_earth) * q / 4    # собственное ИК Земли
q_earth_alb = phi * Albedo_earth * q             # отражённый солнечный свет