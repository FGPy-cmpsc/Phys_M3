from __future__ import annotations
import argparse
import json
import numpy as np

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D

from config import SimulationConfig
from field import field_on_axis
from solver import compute_trajectory, compute_ensemble


def plot_trajectory_3d(traj, cfg: SimulationConfig, ax):
    ax.plot(traj.x * 1e3, traj.y * 1e3, traj.z, linewidth=0.4, alpha=0.8)
    ax.scatter(traj.x[0]*1e3, traj.y[0]*1e3, traj.z[0],
               color='green', s=30, label='старт')
    ax.scatter(traj.x[-1]*1e3, traj.y[-1]*1e3, traj.z[-1],
               color='red', s=30, label='финиш')

    xmin, xmax = traj.x.min() * 1e3, traj.x.max() * 1e3
    ymin, ymax = traj.y.min() * 1e3, traj.y.max() * 1e3
    pad = max(xmax - xmin, ymax - ymin) * 0.15
    corners_x = [xmin - pad, xmax + pad, xmax + pad, xmin - pad, xmin - pad]
    corners_y = [ymin - pad, ymin - pad, ymax + pad, ymax + pad, ymin - pad]
    for zc in [cfg.coil1_z, cfg.coil2_z]:
        ax.plot(corners_x, corners_y,
                [zc] * 5, 'c-', linewidth=1.5, alpha=0.6,
                label=f'кольцо z={zc:.2f} м' if zc == cfg.coil1_z else None)
    ax.legend(fontsize=6, loc='upper left')

    ax.set_xlabel('x, мм')
    ax.set_ylabel('y, мм')
    ax.set_zlabel('z, м')
    status = 'ВЫЛЕТ' if traj.escaped else 'УДЕРЖАНИЕ'
    ax.set_title(f'Траектория ({status}, α₀={traj.pitch_angle_deg:.1f}°)')


def plot_z_t(traj, cfg, ax):
    ax.plot(traj.t * 1e6, traj.z, linewidth=0.5)
    ax.axhline(cfg.coil1_z, color='cyan', linestyle='--', linewidth=0.8, label='кольца')
    ax.axhline(cfg.coil2_z, color='cyan', linestyle='--', linewidth=0.8)
    ax.set_xlabel('t, мкс')
    ax.set_ylabel('z, м')
    ax.set_title('z(t)')
    ax.legend(fontsize=8)


def plot_energy_conservation(traj, cfg, ax):
    v2 = traj.vx**2 + traj.vy**2 + traj.vz**2
    E_kin = 0.5 * cfg.particle_mass * v2
    E0 = E_kin[0]
    rel_err = (E_kin - E0) / E0

    ax.plot(traj.t * 1e6, rel_err, linewidth=0.5)
    ax.set_xlabel('t, мкс')
    ax.set_ylabel('(E - E₀)/E₀')
    ax.set_title(f'Сохранение энергии (макс. ошибка {np.max(np.abs(rel_err)):.2e})')
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(-2, 2))



def plot_ensemble_results(results: list, alpha_crit: float, ax):
    angles = [r.pitch_angle_deg for r in results]
    escaped = [r.escaped for r in results]

    confined = [a for a, e in zip(angles, escaped) if not e]
    lost = [a for a, e in zip(angles, escaped) if e]

    ax.hist(confined, bins=30, range=(0, 90), alpha=0.7,
            color='green', label=f'Удержание ({len(confined)})')
    ax.hist(lost, bins=30, range=(0, 90), alpha=0.7,
            color='red', label=f'Вылет ({len(lost)})')
    ax.axvline(alpha_crit, color='blue', linestyle='--', linewidth=2,
               label=f'α_crit = {alpha_crit:.1f}° (теория)')
    ax.set_xlabel('Начальный питч-угол α₀, градусы')
    ax.set_ylabel('Число частиц')
    ax.set_title('Условия удержания: ансамбль частиц')
    ax.legend(fontsize=8)


def parse_args():
    p = argparse.ArgumentParser(description='Магнитная пробка')
    p.add_argument('--output', type=str, default='magnetic_trap')
    p.add_argument('--ensemble', type=int, default=0)
    p.add_argument('--config', type=str, default=None)
    p.add_argument('--coil-radius', type=float, default=None)
    p.add_argument('--coil-current', type=float, default=None)
    p.add_argument('--coil-separation', type=float, default=None)
    p.add_argument('--charge', type=float, default=None)
    p.add_argument('--mass', type=float, default=None)
    p.add_argument('--x0', type=float, default=None)
    p.add_argument('--y0', type=float, default=None)
    p.add_argument('--z0', type=float, default=None)
    p.add_argument('--vx0', type=float, default=None)
    p.add_argument('--vy0', type=float, default=None)
    p.add_argument('--vz0', type=float, default=None)
    p.add_argument('--t-max', type=float, default=None)
    p.add_argument('--n-steps', type=int, default=None)
    return p.parse_args()


def build_config(args) -> SimulationConfig:
    if args.config:
        cfg = SimulationConfig.load(args.config)
    else:
        cfg = SimulationConfig()

    overrides = {
        'coil_radius': args.coil_radius,
        'coil_current': args.coil_current,
        'coil_separation': args.coil_separation,
        'particle_charge': args.charge,
        'particle_mass': args.mass,
        'x0': args.x0, 'y0': args.y0, 'z0': args.z0,
        'vx0': args.vx0, 'vy0': args.vy0, 'vz0': args.vz0,
        't_max': args.t_max,
        'n_steps': args.n_steps,
    }
    for k, v in overrides.items():
        if v is not None:
            setattr(cfg, k, v)
    return cfg


def main():
    args = parse_args()
    cfg = build_config(args)

    try:
        cfg.validate()
    except ValueError as e:
        print(e)
        return

    if args.ensemble < 0:
        print('Ошибка: --ensemble должен быть >= 0')
        return

    print('=== Магнитная пробка ===')
    print(f'Радиус колец:     {cfg.coil_radius} м')
    print(f'Ток:              {cfg.coil_current:.2e} А')
    print(f'Расстояние:       {cfg.coil_separation} м')
    print(f'Частица: q={cfg.particle_charge:.3e} Кл, m={cfg.particle_mass:.3e} кг')
    print(f'r0 = ({cfg.x0}, {cfg.y0}, {cfg.z0}) м')
    print(f'v0 = ({cfg.vx0:.2e}, {cfg.vy0:.2e}, {cfg.vz0:.2e}) м/с')
    print(f't_max = {cfg.t_max:.2e} с, шагов = {cfg.n_steps}')
    print()

    print('Расчёт траектории...')
    traj = compute_trajectory(cfg)
    status = 'ВЫЛЕТ' if traj.escaped else 'УДЕРЖАНИЕ'
    print(f'Результат: {status}, питч-угол = {traj.pitch_angle_deg:.1f}°')

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle('Магнитная пробка: одиночная траектория', fontsize=14)
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3,
                  height_ratios=[2, 1])

    ax_3d = fig.add_subplot(gs[0, :], projection='3d')
    plot_trajectory_3d(traj, cfg, ax_3d)

    ax_z = fig.add_subplot(gs[1, 0])
    plot_z_t(traj, cfg, ax_z)

    ax_energy = fig.add_subplot(gs[1, 1])
    plot_energy_conservation(traj, cfg, ax_energy)

    fig.savefig(f'{args.output}_single.png', dpi=150, bbox_inches='tight')
    print(f'Сохранено: {args.output}_single.png')
    plt.close(fig)

    z1, z2 = cfg.coil1_z, cfg.coil2_z
    z_arr = np.linspace(z1, z2, 500)
    B_axis = field_on_axis(z_arr, cfg.coil_radius, cfg.coil_current, z1, z2)
    B_min = np.min(B_axis)
    B_max = np.max(B_axis)
    mirror_ratio = B_max / B_min
    alpha_crit = np.degrees(np.arcsin(np.sqrt(B_min / B_max)))
    print(f'\nЗеркальное отношение R_m = {mirror_ratio:.3f}')
    print(f'Критический питч-угол α_crit = {alpha_crit:.1f}°')

    n_ens = args.ensemble
    if n_ens > 0:
        print(f'\nРасчёт ансамбля из {n_ens} частиц...')
        results = compute_ensemble(cfg, n_particles=n_ens)

        n_confined = sum(1 for r in results if not r.escaped)
        n_lost = sum(1 for r in results if r.escaped)
        print(f'Удержано: {n_confined}/{n_ens}  ({100*n_confined/n_ens:.1f}%)')
        print(f'Вылетело: {n_lost}/{n_ens}  ({100*n_lost/n_ens:.1f}%)')

        fig3, ax_ens = plt.subplots(figsize=(8, 5))
        plot_ensemble_results(results, alpha_crit, ax_ens)
        fig3.tight_layout()

        fig3.savefig(f'{args.output}_ensemble.png', dpi=150, bbox_inches='tight')
        print(f'Сохранено: {args.output}_ensemble.png')
        plt.close(fig3)

        summary = {
            'mirror_ratio': mirror_ratio,
            'alpha_crit_deg': alpha_crit,
            'n_particles': n_ens,
            'n_confined': n_confined,
            'n_lost': n_lost,
            'fraction_confined': n_confined / n_ens,
        }
        with open(f'{args.output}_results.json', 'w') as f:
            json.dump(summary, f, indent=2)
        print(f'Сохранено: {args.output}_results.json')

    print('\nГотово.')


if __name__ == '__main__':
    main()
