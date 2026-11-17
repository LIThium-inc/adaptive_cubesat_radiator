import thermal_model.parameters as par  # type: ignore


def A_eff_of_epsilon(epsilon_eff):
    """Поглощательность радиатора в видимом диапазоне при заданной ε_eff.

    Линейная модель по реальным концам диапазона (см. parameters.py).
    """
    return par.A_eff_at_max + par.c * (epsilon_eff - par.epsilon_eff_max)


def _resolve(S_rad, side):
    """Подставляет значения по умолчанию из parameters.py и проверяет side."""
    if S_rad is None:
        S_rad = par.S_rad
    if side is None:
        side = par.RADIATOR_SIDE
    if side not in ("sun", "earth"):
        raise ValueError(f"RADIATOR_SIDE должен быть 'sun' или 'earth', получено: {side!r}")
    return S_rad, side


def panel_terms(side):
    """Слагаемые, зависящие от ориентации КА: (C_, Q_hot_refl, Q_earth).

    Число панелей, обращённых к Солнцу и к Земле, зависит от того, на какую
    сторону смотрит радиатор (N_PANELS_SUN / N_PANELS_EARTH в parameters.py).
    Если этих словарей в parameters.py нет, используются прежние значения
    C_, Q_hot_refl, Q_earth (одинаковые для обеих сторон).
    """
    if not hasattr(par, "N_PANELS_SUN"):
        return par.C_, par.Q_hot_refl, par.Q_earth

    S_sun = par.N_PANELS_SUN[side] * par.S_panel_one      # панели, смотрящие на Солнце, м^2
    S_earth = par.N_PANELS_EARTH[side] * par.S_panel_one  # панели, смотрящие на Землю, м^2

    C_ = S_sun * par.A_solar                                       # поглощение прямого Солнца
    Q_hot_refl = S_earth * par.A_solar * par.q * par.Albedo_earth * par.phi
    Q_earth = S_earth * par.epsilon_solar * par.phi * ((1 - par.Albedo_earth) * par.q) / 4
    return C_, Q_hot_refl, Q_earth


def Q_in_hot(epsilon_eff, S_rad=None, side=None):
    """Поглощённая мощность в горячем режиме, Вт.

    side = "sun"   - радиатор смотрит на Солнце (прямой солнечный поток на радиатор);
    side = "earth" - радиатор смотрит на Землю (ИК Земли и альбедо вместо прямого Солнца).
    """
    S_rad, side = _resolve(S_rad, side)
    A_eff = A_eff_of_epsilon(epsilon_eff)
    C_, Q_hot_refl, Q_earth = panel_terms(side)

    # Всё, что не зависит от радиатора
    Q = par.q * C_ + Q_hot_refl + Q_earth + par.Q_hot_internal

    if side == "sun":
        Q += par.q * S_rad * A_eff
    else:  # "earth"
        Q += S_rad * (epsilon_eff * par.q_earth_ir + A_eff * par.q_earth_alb)
    return Q


def Q_in_cold(epsilon_eff, S_rad=None, side=None):
    """Поглощённая мощность в холодном режиме (тень), Вт. Альбедо отсутствует."""
    S_rad, side = _resolve(S_rad, side)
    _, _, Q_earth = panel_terms(side)

    Q = Q_earth + par.Q_cold_internal
    if side == "earth":
        Q += S_rad * epsilon_eff * par.q_earth_ir
    return Q


def T_hot_epsilon(epsilon_eff, S_rad=None, side=None):
    """Температура в горячем режиме, К. Работает и с числами, и с массивами numpy."""
    S_rad, side = _resolve(S_rad, side)
    Q_in = Q_in_hot(epsilon_eff, S_rad, side)
    return (Q_in / (par.sigma * (S_rad * epsilon_eff + par.C))) ** 0.25


def T_cold_epsilon(epsilon_eff, S_rad=None, side=None):
    """Температура в холодном режиме, К."""
    S_rad, side = _resolve(S_rad, side)
    Q_in = Q_in_cold(epsilon_eff, S_rad, side)
    return (Q_in / (par.sigma * (S_rad * epsilon_eff + par.C))) ** 0.25