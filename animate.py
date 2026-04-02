from __future__ import annotations
import argparse
import numpy as np

import matplotlib
matplotlib.use('webagg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec

from field import magnetic_field_cartesian


def parse_args():
    p = argparse.ArgumentParser(description='Магнитная пробка: анимация')
    p.add_argument('--coil-radius', type=float, default=0.3)
    p.add_argument('--coil-current', type=float, default=5e5)
    p.add_argument('--coil-separation', type=float, default=0.8)
    p.add_argument('--charge', type=float, default=1.602e-19)
    p.add_argument('--mass', type=float, default=1.673e-27)
    p.add_argument('--x0', type=float, default=0.0)
    p.add_argument('--y0', type=float, default=0.02)
    p.add_argument('--z0', type=float, default=0.0)
    p.add_argument('--vx0', type=float, default=8.2e4)
    p.add_argument('--vy0', type=float, default=0.0)
    p.add_argument('--vz0', type=float, default=5.7e4)
    p.add_argument('--t-max', type=float, default=2e-4)
    p.add_argument('--n-steps', type=int, default=20000)
    p.add_argument('--speed', type=int, default=50)
    p.add_argument('--port', type=int, default=8988)
    return p.parse_args()


def boris_push(v, B, q_over_m, dt):
    t_vec = B * (q_over_m * dt / 2)
    t2 = np.dot(t_vec, t_vec)
    s_vec = 2 * t_vec / (1 + t2)
    v_prime = v + np.cross(v, t_vec)
    return v + np.cross(v_prime, s_vec)


def validate_args(args):
    errors = []
    if args.coil_radius <= 0:
        errors.append(f'coil-radius должен быть > 0, получено {args.coil_radius}')
    if args.coil_current == 0:
        errors.append('coil-current не может быть 0')
    if args.coil_separation <= 0:
        errors.append(f'coil-separation должен быть > 0, получено {args.coil_separation}')
    if args.charge == 0:
        errors.append('charge не может быть 0')
    if args.mass <= 0:
        errors.append(f'mass должна быть > 0, получено {args.mass}')
    if args.t_max <= 0:
        errors.append(f't-max должен быть > 0, получено {args.t_max}')
    if args.n_steps <= 0:
        errors.append(f'n-steps должен быть > 0, получено {args.n_steps}')
    if args.speed <= 0:
        errors.append(f'speed должен быть > 0, получено {args.speed}')
    if not (1024 <= args.port <= 65535):
        errors.append(f'port должен быть в диапазоне 1024-65535, получено {args.port}')
    v_total = np.sqrt(args.vx0**2 + args.vy0**2 + args.vz0**2)
    if v_total == 0:
        errors.append('полная скорость частицы не может быть 0')
    rho0 = np.sqrt(args.x0**2 + args.y0**2)
    if rho0 >= args.coil_radius:
        errors.append(
            f'начальное положение (ρ={rho0:.4f} м) должно быть '
            f'внутри кольца (R={args.coil_radius} м)')
    if errors:
        raise ValueError('Ошибки в параметрах:\n  ' + '\n  '.join(errors))


def main():
    args = parse_args()

    try:
        validate_args(args)
    except ValueError as e:
        print(e)
        return

    matplotlib.rcParams['webagg.port'] = args.port

    R = args.coil_radius
    I = args.coil_current
    d = args.coil_separation
    z1, z2 = -d / 2, d / 2
    q_over_m = args.charge / args.mass
    dt = args.t_max / args.n_steps
    z_limit = 3.0 * d
    steps_per_frame = args.speed

    r = np.array([args.x0, args.y0, args.z0])
    v = np.array([args.vx0, args.vy0, args.vz0])

    xs, ys, zs, ts = [r[0]], [r[1]], [r[2]], [0.0]

    fig = plt.figure(figsize=(14, 6))
    gs = GridSpec(1, 2, figure=fig, wspace=0.3, width_ratios=[1.2, 1])

    ax3d = fig.add_subplot(gs[0, 0], projection='3d')
    ax_zt = fig.add_subplot(gs[0, 1])

    line3d, = ax3d.plot([], [], [], 'b-', linewidth=0.5, alpha=0.8)
    point3d, = ax3d.plot([], [], [], 'ro', markersize=5)

    plane_size = 30
    px = [-plane_size, plane_size, plane_size, -plane_size, -plane_size]
    py = [-plane_size, -plane_size, plane_size, plane_size, -plane_size]
    ax3d.plot(px, py, [z1]*5, 'c-', linewidth=1.5, alpha=0.5,
              label='плоскости колец')
    ax3d.plot(px, py, [z2]*5, 'c-', linewidth=1.5, alpha=0.5)

    ax3d.set_xlim(-plane_size, plane_size)
    ax3d.set_ylim(-plane_size, plane_size)
    ax3d.set_zlim(z1 - 0.2 * d, z2 + 0.2 * d)
    ax3d.set_xlabel('x, мм')
    ax3d.set_ylabel('y, мм')
    ax3d.set_zlabel('z, м')
    ax3d.set_title('Траектория частицы')
    ax3d.legend(fontsize=7, loc='upper left')

    line_zt, = ax_zt.plot([], [], linewidth=0.8)
    ax_zt.axhline(z1, color='cyan', linestyle='--', linewidth=0.8, label='кольца')
    ax_zt.axhline(z2, color='cyan', linestyle='--', linewidth=0.8)
    ax_zt.set_xlabel('t, мкс')
    ax_zt.set_ylabel('z, м')
    ax_zt.set_title('z(t)')
    ax_zt.legend(fontsize=8)
    ax_zt.set_xlim(0, args.t_max * 1e6)
    ax_zt.set_ylim(z1 - 0.3 * d, z2 + 0.3 * d)

    step_count = [0]
    escaped = [False]

    def update(frame):
        nonlocal r, v

        if escaped[0] or step_count[0] >= args.n_steps:
            return line3d, point3d, line_zt

        for _ in range(steps_per_frame):
            if step_count[0] >= args.n_steps:
                break
            Bx, By, Bz = magnetic_field_cartesian(
                r[0], r[1], r[2], R, I, z1, z2)
            B = np.array([float(Bx), float(By), float(Bz)])
            v = boris_push(v, B, q_over_m, dt)
            r = r + v * dt
            step_count[0] += 1

            xs.append(r[0])
            ys.append(r[1])
            zs.append(r[2])
            ts.append(step_count[0] * dt)

            if abs(r[2]) > z_limit:
                escaped[0] = True
                ax3d.set_title('Траектория частицы - ВЫЛЕТ')
                break

        x_mm = [xi * 1e3 for xi in xs]
        y_mm = [yi * 1e3 for yi in ys]
        line3d.set_data_3d(x_mm, y_mm, zs)
        point3d.set_data_3d([x_mm[-1]], [y_mm[-1]], [zs[-1]])

        if escaped[0]:
            zmin_plot = min(min(zs), z1) - 0.1
            zmax_plot = max(max(zs), z2) + 0.1
            ax3d.set_zlim(zmin_plot, zmax_plot)
            ax_zt.set_ylim(zmin_plot, zmax_plot)

        t_us = [ti * 1e6 for ti in ts]
        line_zt.set_data(t_us, zs)

        return line3d, point3d, line_zt

    n_frames = args.n_steps // steps_per_frame + 1
    anim = FuncAnimation(fig, update, frames=n_frames,
                         interval=30, blit=False, repeat=False)

    print(f'Анимация запущена на http://127.0.0.1:{args.port}')
    print('Откройте в браузере. Ctrl+C для остановки.')
    plt.show()


if __name__ == '__main__':
    main()
