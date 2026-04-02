from __future__ import annotations
import os
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from config import SimulationConfig
from field import field_on_axis
from solver import compute_trajectory, compute_ensemble

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'demo_output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def demo_single_particle():
    R = 0.3
    I = 5e5
    d = 0.8
    v_total = 1e5

    z1 = -d / 2
    z2 = d / 2

    B_axis = field_on_axis(np.array([0.0, z1, z2]), R, I, z1, z2)
    B_min = B_axis[0]
    B_max = max(B_axis[1], B_axis[2])
    Rm = B_max / B_min
    alpha_crit = np.degrees(np.arcsin(np.sqrt(1.0 / Rm)))

    print(f'R_m = {Rm:.3f}, α_crit = {alpha_crit:.1f}°')

    alpha_conf = alpha_crit + 15.0
    alpha_conf_rad = np.radians(alpha_conf)
    cfg_conf = SimulationConfig(
        coil_radius=R, coil_current=I, coil_separation=d,
        x0=0.0, y0=0.02, z0=0.0,
        vx0=v_total * np.sin(alpha_conf_rad),
        vy0=0.0,
        vz0=v_total * np.cos(alpha_conf_rad),
        t_max=2e-4, n_steps=20000,
    )

    alpha_esc = max(alpha_crit - 15.0, 3.0)
    alpha_esc_rad = np.radians(alpha_esc)
    cfg_esc = SimulationConfig(
        coil_radius=R, coil_current=I, coil_separation=d,
        x0=0.0, y0=0.02, z0=0.0,
        vx0=v_total * np.sin(alpha_esc_rad),
        vy0=0.0,
        vz0=v_total * np.cos(alpha_esc_rad),
        t_max=2e-4, n_steps=20000,
    )

    print('Расчёт удерживаемой частицы...')
    traj_conf = compute_trajectory(cfg_conf)
    print(f'  α₀ = {traj_conf.pitch_angle_deg:.1f}°, '
          f'escaped = {traj_conf.escaped}')

    print('Расчёт вылетающей частицы...')
    traj_esc = compute_trajectory(cfg_esc)
    print(f'  α₀ = {traj_esc.pitch_angle_deg:.1f}°, '
          f'escaped = {traj_esc.escaped}')

    fig = plt.figure(figsize=(16, 18))
    fig.suptitle('Магнитная пробка: одиночные траектории', fontsize=14)
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3,
                  height_ratios=[2, 1, 1])

    ax_3d_c = fig.add_subplot(gs[0, 0], projection='3d')
    _plot_3d_zoomed(traj_conf, cfg_conf, ax_3d_c, 'Удержание')

    ax_3d_e = fig.add_subplot(gs[0, 1], projection='3d')
    _plot_3d_zoomed(traj_esc, cfg_esc, ax_3d_e, 'Вылет')

    ax_z_c = fig.add_subplot(gs[1, 0])
    ax_z_c.plot(traj_conf.t * 1e6, traj_conf.z, linewidth=0.5)
    ax_z_c.axhline(z1, color='cyan', linestyle='--', linewidth=0.8, label='кольца')
    ax_z_c.axhline(z2, color='cyan', linestyle='--', linewidth=0.8)
    ax_z_c.set_xlabel('t, мкс')
    ax_z_c.set_ylabel('z, м')
    ax_z_c.set_title(f'z(t): удержание (α₀={traj_conf.pitch_angle_deg:.1f}°)')
    ax_z_c.legend(fontsize=8)

    ax_z_e = fig.add_subplot(gs[1, 1])
    ax_z_e.plot(traj_esc.t * 1e6, traj_esc.z, linewidth=0.5, color='tab:orange')
    ax_z_e.axhline(z1, color='cyan', linestyle='--', linewidth=0.8, label='кольца')
    ax_z_e.axhline(z2, color='cyan', linestyle='--', linewidth=0.8)
    ax_z_e.set_xlabel('t, мкс')
    ax_z_e.set_ylabel('z, м')
    ax_z_e.set_title(f'z(t): вылет (α₀={traj_esc.pitch_angle_deg:.1f}°)')
    ax_z_e.legend(fontsize=8)

    ax_en_c = fig.add_subplot(gs[2, 0])
    v2_c = traj_conf.vx**2 + traj_conf.vy**2 + traj_conf.vz**2
    E_c = 0.5 * cfg_conf.particle_mass * v2_c
    rel_c = (E_c - E_c[0]) / E_c[0]
    ax_en_c.plot(traj_conf.t * 1e6, rel_c, linewidth=0.5)
    ax_en_c.set_xlabel('t, мкс')
    ax_en_c.set_ylabel('(E-E₀)/E₀')
    ax_en_c.set_title(f'Сохранение энергии: удержание (макс. {np.max(np.abs(rel_c)):.2e})')
    ax_en_c.ticklabel_format(style='scientific', axis='y', scilimits=(-2, 2))

    ax_en_e = fig.add_subplot(gs[2, 1])
    v2_e = traj_esc.vx**2 + traj_esc.vy**2 + traj_esc.vz**2
    E_e = 0.5 * cfg_esc.particle_mass * v2_e
    rel_e = (E_e - E_e[0]) / E_e[0]
    ax_en_e.plot(traj_esc.t * 1e6, rel_e, linewidth=0.5, color='tab:orange')
    ax_en_e.set_xlabel('t, мкс')
    ax_en_e.set_ylabel('(E-E₀)/E₀')
    ax_en_e.set_title(f'Сохранение энергии: вылет (макс. {np.max(np.abs(rel_e)):.2e})')
    ax_en_e.ticklabel_format(style='scientific', axis='y', scilimits=(-2, 2))

    path = os.path.join(OUTPUT_DIR, 'single_particles.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Сохранено: {path}')
    return cfg_conf, Rm, alpha_crit


def demo_ensemble(cfg_base: SimulationConfig, alpha_crit: float):
    N = 50
    cfg_ens = SimulationConfig(
        coil_radius=cfg_base.coil_radius,
        coil_current=cfg_base.coil_current,
        coil_separation=cfg_base.coil_separation,
        particle_charge=cfg_base.particle_charge,
        particle_mass=cfg_base.particle_mass,
        x0=cfg_base.x0, y0=cfg_base.y0, z0=cfg_base.z0,
        vx0=cfg_base.vx0, vy0=cfg_base.vy0, vz0=cfg_base.vz0,
        t_max=5e-5, n_steps=5000,
    )
    print(f'\nРасчёт ансамбля из {N} частиц...')
    results = compute_ensemble(cfg_ens, n_particles=N)

    n_conf = sum(1 for r in results if not r.escaped)
    n_lost = sum(1 for r in results if r.escaped)
    print(f'Удержано: {n_conf}/{N}  ({100*n_conf/N:.1f}%)')
    print(f'Вылетело: {n_lost}/{N}  ({100*n_lost/N:.1f}%)')

    frac_theory = np.cos(np.radians(alpha_crit))
    print(f'Теоретическая доля удержания: {(1-frac_theory)*100:.1f}%')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Магнитная пробка: ансамбль частиц', fontsize=13)

    ax = axes[0]
    angles_conf = [r.pitch_angle_deg for r in results if not r.escaped]
    angles_lost = [r.pitch_angle_deg for r in results if r.escaped]
    ax.hist(angles_conf, bins=30, range=(0, 90), alpha=0.7,
            color='green', label=f'Удержание ({n_conf})')
    ax.hist(angles_lost, bins=30, range=(0, 90), alpha=0.7,
            color='red', label=f'Вылет ({n_lost})')
    ax.axvline(alpha_crit, color='blue', linestyle='--', linewidth=2,
               label=f'α_crit = {alpha_crit:.1f}°')
    ax.set_xlabel('Начальный питч-угол α₀, °')
    ax.set_ylabel('Число частиц')
    ax.legend(fontsize=9)

    ax2 = axes[1]
    for r in results:
        color = 'green' if not r.escaped else 'red'
        z_max = np.max(np.abs(r.z))
        ax2.scatter(r.pitch_angle_deg, z_max, c=color, s=8, alpha=0.5)
    ax2.axvline(alpha_crit, color='blue', linestyle='--', linewidth=2,
                label=f'α_crit = {alpha_crit:.1f}°')
    ax2.axhline(abs(cfg_base.coil1_z), color='cyan', linestyle=':',
                label='Положение колец')
    ax2.set_xlabel('Начальный питч-угол α₀, °')
    ax2.set_ylabel('max |z|, м')
    ax2.set_title('Максимальное отклонение по z')
    ax2.legend(fontsize=9)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'ensemble.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Сохранено: {path}')


def _plot_3d_zoomed(traj, cfg, ax, label):
    ax.plot(traj.x * 1e3, traj.y * 1e3, traj.z, linewidth=0.3, alpha=0.8)
    ax.scatter(traj.x[0]*1e3, traj.y[0]*1e3, traj.z[0], color='green', s=20)
    ax.scatter(traj.x[-1]*1e3, traj.y[-1]*1e3, traj.z[-1], color='red', s=20)

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
    ax.set_title(f'{label} (α₀={traj.pitch_angle_deg:.1f}°)')


if __name__ == '__main__':
    cfg_base, Rm, alpha_crit = demo_single_particle()
    demo_ensemble(cfg_base, alpha_crit)
    print('\nДемонстрация завершена.')
