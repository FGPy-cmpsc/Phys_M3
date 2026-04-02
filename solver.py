from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from config import SimulationConfig
from field import magnetic_field_cartesian


@dataclass
class TrajectoryResult:
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    vz: np.ndarray
    escaped: bool
    pitch_angle_deg: float


def _boris_push(v, B, q_over_m, dt):
    t_vec = B * (q_over_m * dt / 2)
    t2 = np.dot(t_vec, t_vec)
    s_vec = 2 * t_vec / (1 + t2)
    v_prime = v + np.cross(v, t_vec)
    return v + np.cross(v_prime, s_vec)


def compute_trajectory(cfg: SimulationConfig) -> TrajectoryResult:
    q_over_m = cfg.particle_charge / cfg.particle_mass
    z1, z2 = cfg.coil1_z, cfg.coil2_z
    R, I = cfg.coil_radius, cfg.coil_current
    dt = cfg.t_max / cfg.n_steps
    z_limit = 3.0 * cfg.coil_separation

    r = np.array([cfg.x0, cfg.y0, cfg.z0])
    v = np.array([cfg.vx0, cfg.vy0, cfg.vz0])

    v_perp = np.sqrt(cfg.vx0**2 + cfg.vy0**2)
    v_par = abs(cfg.vz0)
    v_total = np.sqrt(v_perp**2 + v_par**2)
    pitch_angle = np.degrees(np.arctan2(v_perp, v_par)) if v_total > 0 else 0.0

    xs = np.empty(cfg.n_steps + 1)
    ys = np.empty(cfg.n_steps + 1)
    zs = np.empty(cfg.n_steps + 1)
    vxs = np.empty(cfg.n_steps + 1)
    vys = np.empty(cfg.n_steps + 1)
    vzs = np.empty(cfg.n_steps + 1)
    ts = np.empty(cfg.n_steps + 1)

    xs[0], ys[0], zs[0] = r
    vxs[0], vys[0], vzs[0] = v
    ts[0] = 0.0

    escaped = False
    last_idx = cfg.n_steps

    for i in range(cfg.n_steps):
        Bx, By, Bz = magnetic_field_cartesian(
            r[0], r[1], r[2], R, I, z1, z2)
        B = np.array([float(Bx), float(By), float(Bz)])

        v = _boris_push(v, B, q_over_m, dt)
        r = r + v * dt

        idx = i + 1
        xs[idx], ys[idx], zs[idx] = r
        vxs[idx], vys[idx], vzs[idx] = v
        ts[idx] = idx * dt

        if abs(r[2]) > z_limit:
            escaped = True
            last_idx = idx
            break

    s = slice(0, last_idx + 1)
    return TrajectoryResult(
        t=ts[s], x=xs[s], y=ys[s], z=zs[s],
        vx=vxs[s], vy=vys[s], vz=vzs[s],
        escaped=escaped,
        pitch_angle_deg=pitch_angle
    )


def compute_ensemble(cfg: SimulationConfig, n_particles: int = 200,
                     seed: int = 42) -> list[TrajectoryResult]:
    rng = np.random.default_rng(seed)
    results = []

    v_total = np.sqrt(cfg.vx0**2 + cfg.vy0**2 + cfg.vz0**2)
    if v_total == 0:
        v_total = 1e5

    n_steps_ens = min(cfg.n_steps, 2000)

    for i in range(n_particles):
        pitch = rng.uniform(0, 90)
        phi = rng.uniform(0, 2 * np.pi)

        pitch_rad = np.radians(pitch)
        v_perp = v_total * np.sin(pitch_rad)
        v_par = v_total * np.cos(pitch_rad)

        cfg_i = SimulationConfig(
            coil_radius=cfg.coil_radius,
            coil_current=cfg.coil_current,
            coil_separation=cfg.coil_separation,
            particle_charge=cfg.particle_charge,
            particle_mass=cfg.particle_mass,
            x0=cfg.x0, y0=cfg.y0, z0=cfg.z0,
            vx0=v_perp * np.cos(phi),
            vy0=v_perp * np.sin(phi),
            vz0=v_par,
            t_max=cfg.t_max,
            n_steps=n_steps_ens,
        )
        results.append(compute_trajectory(cfg_i))

    return results
