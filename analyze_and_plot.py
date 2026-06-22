import numpy as np
from scipy.interpolate import RegularGridInterpolator
import os
from datetime import datetime
import matplotlib.pyplot as plt
import pickle

# --- 1. Load Cached Data ---
DATA_CACHE_FILE = "400nm_ring_400nm_wg_556nm_light_175nm_gap.pkl"

print(f"Loading simulation data from {DATA_CACHE_FILE}...")
with open(DATA_CACHE_FILE, 'rb') as f:
    saved_data = pickle.load(f)
    ring_data = saved_data['ring_data']
    wg_data = saved_data['wg_data']
    wavelengths = saved_data['wavelengths']
    params = saved_data['params']

R = params['R']
w_R = params['W_r']
w_wg = params['W_wg']
G = params['G']
c0 = params['c0']
R_wg = R + (0.5*w_R) + G + (0.5*w_wg)

PLOT_DIR = "cmt_diagnostics"
if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)

# --- 2. Core CMT Math Functions ---
def compute_cmt_base(idx):
    """Calculates Gamma and delta_beta for a specific wavelength index."""
    wl = wavelengths[idx]
    omega = 2 * np.pi * c0 / wl
    
    E_R = ring_data['E'][idx]
    eps_R = ring_data['eps'][idx]
    neff_R = np.real(ring_data['neff'][idx])
    ng_R = np.squeeze(np.real(ring_data['ng'][idx]))
    
    E_wg = wg_data['E'][idx]
    eps_wg = wg_data['eps'][idx]
    neff_wg = np.real(wg_data['neff'][idx])
    
    # Grid Alignment
    offset_x = R_wg - R 
    wg_x_shifted = wg_data['x'] + offset_x
    
    common_x = np.linspace(min(ring_data['x'][0], wg_x_shifted[0]), max(ring_data['x'][-1], wg_x_shifted[-1]), 500)
    common_y = np.linspace(min(ring_data['y'][0], wg_data['y'][0]), max(ring_data['y'][-1], wg_data['y'][-1]), 500)
    X, Y = np.meshgrid(common_x, common_y, indexing='ij')
    pts = np.array([X.flatten(), Y.flatten()]).T
    
    # Interpolation
    interp_E_R = RegularGridInterpolator((ring_data['x'], ring_data['y']), E_R, bounds_error=False, fill_value=0)
    interp_E_wg = RegularGridInterpolator((wg_x_shifted, wg_data['y']), E_wg, bounds_error=False, fill_value=0)
    interp_eps_R = RegularGridInterpolator((ring_data['x'], ring_data['y']), eps_R, method='nearest', bounds_error=False, fill_value=None) 
    interp_eps_wg = RegularGridInterpolator((wg_x_shifted, wg_data['y']), eps_wg, method='nearest', bounds_error=False, fill_value=None)
    
    E_R_grid = interp_E_R(pts).reshape((len(common_x), len(common_y), 3))
    E_wg_grid = interp_E_wg(pts).reshape((len(common_x), len(common_y), 3))
    eps_R_grid = interp_eps_R(pts).reshape((len(common_x), len(common_y)))
    eps_wg_grid = interp_eps_wg(pts).reshape((len(common_x), len(common_y)))
    
    # Math
    delta_beta = (omega / c0) * (neff_R * (R / R_wg) - neff_wg)
    dx = common_x[1] - common_x[0]
    dy = common_y[1] - common_y[0]

    eta0 = 376.730313668  
    P_R = (neff_R / (2 * eta0)) * np.sum(np.abs(E_R_grid)**2) * dx * dy
    P_wg = (neff_wg / (2 * eta0)) * np.sum(np.abs(E_wg_grid)**2) * dx * dy

    E_R_grid_norm = E_R_grid / np.sqrt(P_R)
    E_wg_grid_norm = E_wg_grid / np.sqrt(P_wg)

    eps0 = 8.854187817e-12  
    dot_product = np.sum(np.conj(E_R_grid_norm) * E_wg_grid_norm, axis=2) 
    integrand = eps0 * (eps_wg_grid - eps_R_grid) * dot_product

    Gamma = (1j * omega / 4) * np.sum(integrand) * dx * dy
    return Gamma, delta_beta, ng_R, omega

def compute_qc(omega, Gamma, delta_beta, ng_R, angle_deg):
    """Calculates Qc rapidly for any angle using precomputed base parameters."""
    L_c = 2 * np.pi * R * (angle_deg / 360)
    Gamma_mag = np.abs(Gamma)
    
    sinc_arg = L_c * np.sqrt((delta_beta/2)**2 + Gamma_mag**2)
    sinc_val = np.sin(sinc_arg) / sinc_arg 
    
    coupling_factor = (Gamma_mag * L_c * sinc_val)**(-2)
    Qc = omega * (ng_R / c0) * 2 * np.pi * R * coupling_factor
    return Qc

# --- 3. Executable Analyses ---

# Analysis A: Wavelength Sweep (Fixed Angle)
def run_wavelength_sweep(target_angle_deg):
    print(f"\n--- Running Wavelength Sweep at {target_angle_deg} degrees ---")
    Qc_results = []
    
    for i, wl in enumerate(wavelengths):
        Gamma, delta_beta, ng_R, omega = compute_cmt_base(i)
        Qc = compute_qc(omega, Gamma, delta_beta, ng_R, target_angle_deg)
        Qc_results.append(Qc)
        print(f"Wavelength: {wl*1e9:.0f} nm -> Qc: {Qc:.2e}")
    
    # Plotting
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(wavelengths * 1e9, Qc_results, linestyle=':', linewidth=1.2, color='purple')
    ax.scatter(wavelengths * 1e9, Qc_results, color='purple', s=28)
    ax.set_yscale('log') # Log scale is often better for Qc
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Coupling Quality Factor (Qc)")
    ax.set_title(f"Qc vs Wavelength (Angle = {target_angle_deg}°)")
    ax.grid(True, which="both", ls="--", alpha=0.5)
    
    filename = os.path.join(PLOT_DIR, f"qc_vs_wl_{target_angle_deg}deg_{timestamp}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved to {filename}")

# Analysis B: Angle Sweep (Fixed Wavelength)
def run_angle_sweep(target_wl_nm, min_angle, max_angle, num_points):
    print(f"\n--- Running Angle Sweep at {target_wl_nm} nm ---")
    
    # Find the closest wavelength index in our extracted data
    idx = (np.abs(wavelengths - target_wl_nm*1e-9)).argmin()
    actual_wl = wavelengths[idx]
    print(f"Using closest cached wavelength: {actual_wl*1e9:.1f} nm")
    
    # Compute base parameters ONCE for this wavelength
    Gamma, delta_beta, ng_R, omega = compute_cmt_base(idx)
    
    # Sweep the angle
    angles = np.linspace(min_angle, max_angle, num_points)
    Qc_results = [compute_qc(omega, Gamma, delta_beta, ng_R, angle) for angle in angles]
    print('lowest achieved coupling Q is', min(Qc_results), 'at angle', angles[Qc_results.index(min(Qc_results))])
    
    # Plotting
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(angles, Qc_results, linewidth=1.5, color='teal')
    ax.set_yscale('log')
    ax.set_xlabel("Coupling Angle (Degrees)")
    ax.set_ylabel("Coupling Quality Factor (Qc)")
    ax.set_title(f"Qc vs Coupling Angle (Wavelength = {actual_wl*1e9:.1f} nm)")
    ax.grid(True, which="both", ls="--", alpha=0.5)
    
    filename = os.path.join(PLOT_DIR, f"qc_vs_angle_wl_{actual_wl*1e9:.1f}nm_gap_{G*1e9:.1f}_nm_{timestamp}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved to {filename}")

# --- 4. Main Execution ---
if __name__ == "__main__":
    # You can comment/uncomment these to run whichever analysis you need today!
    
    # 1. Sweep wavelength from 555-579nm at 60 degrees
    #run_wavelength_sweep(target_angle_deg=60)
    
    # 2. Sweep angle from 10 to 180 degrees at 556 nm
    run_angle_sweep(target_wl_nm=556.0, min_angle=10, max_angle=15, num_points=200)