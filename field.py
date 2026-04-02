from __future__ import annotations
import numpy as np
from scipy.special import ellipk, ellipe

MU_0 = 4 * np.pi * 1e-7


def _coil_field_cylindrical(rho: np.ndarray, z: np.ndarray,
                            R: float, I: float, z0: float):
    dz = z - z0
    rho = np.asarray(rho, dtype=float)
    dz = np.asarray(dz, dtype=float)

    alpha2 = (R + rho)**2 + dz**2
    beta2 = (R - rho)**2 + dz**2

    with np.errstate(divide='ignore', invalid='ignore'):
        k2 = 1.0 - beta2 / alpha2

    k2 = np.clip(k2, 0.0, 1.0 - 1e-15)

    K = ellipk(k2)
    E = ellipe(k2)

    C = MU_0 * I / (2 * np.pi)
    sqrt_alpha2 = np.sqrt(alpha2)

    with np.errstate(divide='ignore', invalid='ignore'):
        B_z = C / sqrt_alpha2 * (K + (R**2 - rho**2 - dz**2) / beta2 * E)

    with np.errstate(divide='ignore', invalid='ignore'):
        B_rho = C * dz / (rho * sqrt_alpha2) * (-K + (R**2 + rho**2 + dz**2) / beta2 * E)

    on_axis = rho < 1e-12
    B_rho = np.where(on_axis, 0.0, B_rho)

    return B_rho, B_z


def magnetic_field_cartesian(x, y, z, R: float, I: float,
                              z1: float, z2: float):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

    rho = np.sqrt(x**2 + y**2)

    Br1, Bz1 = _coil_field_cylindrical(rho, z, R, I, z1)
    Br2, Bz2 = _coil_field_cylindrical(rho, z, R, I, z2)

    B_rho = Br1 + Br2
    B_z = Bz1 + Bz2

    with np.errstate(divide='ignore', invalid='ignore'):
        cos_phi = np.where(rho > 1e-12, x / rho, 0.0)
        sin_phi = np.where(rho > 1e-12, y / rho, 0.0)

    Bx = B_rho * cos_phi
    By = B_rho * sin_phi

    return Bx, By, B_z


def field_on_axis(z, R: float, I: float, z1: float, z2: float):
    z = np.asarray(z, dtype=float)

    def _single(z_arr, z0):
        dz = z_arr - z0
        return MU_0 * I * R**2 / (2 * (R**2 + dz**2)**1.5)

    return _single(z, z1) + _single(z, z2)
