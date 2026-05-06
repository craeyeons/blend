"""
Compute RMSE of PINN predictions vs CFD ground truth for the NSE models in
./models/:

    - pinn_cylinder_1.0.h5     (cylinder, Re=1)
    - pinn_cylinder_100.0.h5   (cylinder, Re=100)
    - pinn_cavity_flow.h5      (lid-driven cavity, Re=100)

For each model the CFD reference is computed with the same iterative solver
used in plot_coverage_metrics{,_cavity}.py and cached to .npz so reruns are
cheap. RMSE is reported per-field (u, v, p) and as the combined velocity
field over fluid grid points. Pressure is mean-subtracted on the fluid mask
before scoring (gauge invariance).

Usage:
    python compute_nse_rmse.py [--out rmse_summary.txt] [--cfd-dir ./cfd_cache]
"""

import argparse
import os
import time
import numpy as np
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass

from lib.router import create_cylinder_setup, create_cavity_setup
from lib.cylinder_flow import CylinderFlowSimulation
from lib.cavity_flow import CavityFlowSimulation
from cylinder_network import Network as CylinderNetwork
from lib.network import Network as CavityNetwork


def rmse(pred, true, mask):
    m = mask > 0
    diff = (pred - true)[m]
    return float(np.sqrt(np.mean(diff ** 2)))


def gauge_shift(p, mask):
    m = mask > 0
    return p - float(np.mean(p[m]))


def load_or_compute_cfd_cylinder(args, cache_path):
    if os.path.exists(cache_path):
        d = np.load(cache_path)
        print(f"  Loaded cached CFD: {cache_path}")
        return d['u'], d['v'], d['p']
    sim = CylinderFlowSimulation(
        Re=args['Re'], N=args['ny'],
        max_iter=args['max_iter'], tol=args['tol'],
        x_domain=(args['x_min'], args['x_max']),
        y_domain=(args['y_min'], args['y_max']),
        cylinder_center=(args['cylinder_x'], args['cylinder_y']),
        cylinder_radius=args['cylinder_radius'],
        inlet_velocity=args['inlet_velocity'],
    )
    t0 = time.time()
    u, v, p = sim.solve()
    print(f"  CFD solved in {time.time() - t0:.1f}s")
    np.savez(cache_path, u=u, v=v, p=p, X=sim.X, Y=sim.Y)
    return u, v, p


def load_or_compute_cfd_cavity(args, cache_path):
    if os.path.exists(cache_path):
        d = np.load(cache_path)
        print(f"  Loaded cached CFD: {cache_path}")
        return d['u'], d['v'], d['p']
    sim = CavityFlowSimulation(
        Re=args['Re'], N=args['N'],
        max_iter=args['max_iter'], tol=args['tol'],
    )
    t0 = time.time()
    u, v, p = sim.solve()
    print(f"  CFD solved in {time.time() - t0:.1f}s")
    np.savez(cache_path, u=u, v=v, p=p, X=sim.X, Y=sim.Y)
    return u, v, p


def pinn_cylinder(model_path, X, Y, layout):
    net = CylinderNetwork()
    model = net.build(num_inputs=2, layers=[48, 48, 48, 48],
                      activation='tanh', num_outputs=3)
    model.load_weights(model_path)
    xy = np.stack([X.flatten(), Y.flatten()], axis=-1).astype(np.float32)
    out = model.predict(xy, batch_size=len(xy), verbose=0)
    u = out[:, 0].reshape(X.shape) * layout
    v = out[:, 1].reshape(X.shape) * layout
    p = out[:, 2].reshape(X.shape) * layout
    return u.astype(np.float32), v.astype(np.float32), p.astype(np.float32)


def pinn_cavity(model_path, X, Y, layout):
    net = CavityNetwork()
    model = net.build(num_inputs=2, layers=[32, 16, 16, 32],
                      activation='tanh', num_outputs=2)
    model.load_weights(model_path)
    xy = np.stack([X.flatten(), Y.flatten()], axis=-1).astype(np.float32)
    xy_tf = tf.Variable(xy)
    with tf.GradientTape() as tape:
        psi_p = model(xy_tf)
        psi = psi_p[:, 0]
    grad_psi = tape.gradient(psi, xy_tf).numpy()
    u = grad_psi[:, 1].reshape(X.shape) * layout
    v = -grad_psi[:, 0].reshape(X.shape) * layout
    p = psi_p[:, 1].numpy().reshape(X.shape)
    p = (p - p[0, 0]) * layout
    return u.astype(np.float32), v.astype(np.float32), p.astype(np.float32)


def evaluate(name, u_pinn, v_pinn, p_pinn, u_cfd, v_cfd, p_cfd, mask):
    p_pinn_g = gauge_shift(p_pinn, mask)
    p_cfd_g = gauge_shift(p_cfd, mask)
    rmse_u = rmse(u_pinn, u_cfd, mask)
    rmse_v = rmse(v_pinn, v_cfd, mask)
    rmse_p = rmse(p_pinn_g, p_cfd_g, mask)
    diff_vel2 = ((u_pinn - u_cfd) ** 2 + (v_pinn - v_cfd) ** 2)[mask > 0]
    rmse_vel = float(np.sqrt(np.mean(diff_vel2)))
    print(f"  RMSE u   : {rmse_u:.6f}")
    print(f"  RMSE v   : {rmse_v:.6f}")
    print(f"  RMSE p   : {rmse_p:.6f}  (mean-shifted)")
    print(f"  RMSE |U| : {rmse_vel:.6f}")
    return dict(model=name, rmse_u=rmse_u, rmse_v=rmse_v,
                rmse_p=rmse_p, rmse_vel=rmse_vel)


def run_cylinder(model_path, Re, cfd_dir):
    print(f"\n=== Cylinder Re={Re}: {model_path} ===")
    cfg = dict(Re=Re, nx=200, ny=100,
               x_min=0.0, x_max=2.0, y_min=0.0, y_max=1.0,
               cylinder_x=0.5, cylinder_y=0.5, cylinder_radius=0.1,
               inlet_velocity=1.0, max_iter=200000, tol=1e-6)
    X, Y, layout, *_ = create_cylinder_setup(
        Nx=cfg['nx'], Ny=cfg['ny'],
        x_domain=(cfg['x_min'], cfg['x_max']),
        y_domain=(cfg['y_min'], cfg['y_max']),
        cylinder_center=(cfg['cylinder_x'], cfg['cylinder_y']),
        cylinder_radius=cfg['cylinder_radius'],
        inlet_velocity=cfg['inlet_velocity'],
    )
    cache = os.path.join(cfd_dir, f"cfd_cylinder_Re{Re}.npz")
    u_cfd, v_cfd, p_cfd = load_or_compute_cfd_cylinder(cfg, cache)
    u_p, v_p, p_p = pinn_cylinder(model_path, X, Y, layout)
    return evaluate(f"cylinder_Re{Re}", u_p, v_p, p_p,
                    u_cfd, v_cfd, p_cfd, layout)


def run_cavity(model_path, cfd_dir):
    print(f"\n=== Cavity: {model_path} ===")
    cfg = dict(Re=100, N=100, max_iter=200000, tol=1e-6)
    X, Y, layout, *_ = create_cavity_setup(N=cfg['N'])
    cache = os.path.join(cfd_dir, "cfd_cavity_Re100.npz")
    u_cfd, v_cfd, p_cfd = load_or_compute_cfd_cavity(cfg, cache)
    u_p, v_p, p_p = pinn_cavity(model_path, X, Y, layout)
    return evaluate("cavity_Re100", u_p, v_p, p_p,
                    u_cfd, v_cfd, p_cfd, layout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models-dir', default='./models')
    ap.add_argument('--cfd-dir', default='./cfd_cache')
    ap.add_argument('--out', default='rmse_summary.txt')
    args = ap.parse_args()

    os.makedirs(args.cfd_dir, exist_ok=True)

    results = []
    results.append(run_cylinder(
        os.path.join(args.models_dir, 'pinn_cylinder_1.0.h5'), 1.0, args.cfd_dir))
    results.append(run_cylinder(
        os.path.join(args.models_dir, 'pinn_cylinder_100.0.h5'), 100.0, args.cfd_dir))
    results.append(run_cavity(
        os.path.join(args.models_dir, 'pinn_cavity_flow.h5'), args.cfd_dir))

    print("\n" + "=" * 60)
    print(f"{'model':<20s} {'RMSE u':>10s} {'RMSE v':>10s} {'RMSE p':>10s} {'RMSE |U|':>10s}")
    print("-" * 60)
    lines = [f"{'model':<20s} {'RMSE u':>10s} {'RMSE v':>10s} {'RMSE p':>10s} {'RMSE |U|':>10s}"]
    for r in results:
        line = (f"{r['model']:<20s} {r['rmse_u']:>10.6f} {r['rmse_v']:>10.6f} "
                f"{r['rmse_p']:>10.6f} {r['rmse_vel']:>10.6f}")
        print(line)
        lines.append(line)
    with open(args.out, 'w') as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSaved summary to {args.out}")


if __name__ == "__main__":
    main()
