from __future__ import annotations
import json
import numpy as np
from dataclasses import dataclass


@dataclass
class SimulationConfig:
    coil_radius: float = 0.5
    coil_current: float = 1e6
    coil_separation: float = 1.0

    particle_charge: float = 1.602e-19
    particle_mass: float = 1.673e-27

    x0: float = 0.0
    y0: float = 0.05
    z0: float = 0.0
    vx0: float = 0.0
    vy0: float = 0.0
    vz0: float = 1e5

    t_max: float = 1e-4
    n_steps: int = 50000

    def validate(self) -> None:
        errors = []

        if self.coil_radius <= 0:
            errors.append(f'coil_radius должен быть > 0, получено {self.coil_radius}')
        if self.coil_current == 0:
            errors.append('coil_current не может быть 0')
        if self.coil_separation <= 0:
            errors.append(f'coil_separation должен быть > 0, получено {self.coil_separation}')
        if self.particle_charge == 0:
            errors.append('particle_charge не может быть 0')
        if self.particle_mass <= 0:
            errors.append(f'particle_mass должна быть > 0, получено {self.particle_mass}')
        if self.t_max <= 0:
            errors.append(f't_max должен быть > 0, получено {self.t_max}')
        if self.n_steps <= 0:
            errors.append(f'n_steps должен быть > 0, получено {self.n_steps}')

        v_total = np.sqrt(self.vx0**2 + self.vy0**2 + self.vz0**2)
        if v_total == 0:
            errors.append('полная скорость частицы не может быть 0')

        rho0 = np.sqrt(self.x0**2 + self.y0**2)
        if self.coil_radius > 0 and rho0 >= self.coil_radius:
            errors.append(
                f'начальное положение (ρ={rho0:.4f} м) должно быть '
                f'внутри кольца (R={self.coil_radius} м)')

        if self.coil_separation > 0 and abs(self.z0) > self.coil_separation / 2:
            errors.append(
                f'начальное z0={self.z0} м выходит за пределы ловушки '
                f'(±{self.coil_separation/2} м)')

        if not np.isfinite(self.coil_current):
            errors.append('coil_current должен быть конечным числом')
        for name in ['vx0', 'vy0', 'vz0', 'x0', 'y0', 'z0']:
            val = getattr(self, name)
            if not np.isfinite(val):
                errors.append(f'{name} должен быть конечным числом')

        if errors:
            raise ValueError('Ошибки в параметрах:\n  ' + '\n  '.join(errors))

    @property
    def coil1_z(self) -> float:
        return -self.coil_separation / 2

    @property
    def coil2_z(self) -> float:
        return self.coil_separation / 2

    @property
    def r0(self) -> np.ndarray:
        return np.array([self.x0, self.y0, self.z0])

    @property
    def v0(self) -> np.ndarray:
        return np.array([self.vx0, self.vy0, self.vz0])

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()
                if not k.startswith('_')}

    @classmethod
    def from_dict(cls, d: dict) -> SimulationConfig:
        return cls(**{k: v for k, v in d.items()
                      if k in cls.__dataclass_fields__})

    def save(self, path: str) -> None:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> SimulationConfig:
        with open(path, 'r', encoding='utf-8') as f:
            return cls.from_dict(json.load(f))
