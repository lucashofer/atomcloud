ERBIUM66_MASS = 166 * 1.660539e-27
KB = 1.38064852e-23

# __all__ = ["cloud_temperature"]


def cloud_temperature(
    sigma: float, mass: float, tof: float, trap_freq: float = None
) -> float:
    if trap_freq is None:
        return (sigma**2 * mass) / (KB * tof**2)
    else:
        num = sigma**2 * mass * trap_freq**2
        denom = KB * (1 + (trap_freq * tof) ** 2)
        return num / denom
