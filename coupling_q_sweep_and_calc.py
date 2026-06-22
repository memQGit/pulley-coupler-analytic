import numpy as np
from scipy.interpolate import RegularGridInterpolator
import importlib.util
import os
from datetime import datetime
import argparse
import matplotlib.pyplot as plt
import pickle

# # --- 1. Initialize Lumerical API ---
# lumapi_path = r"C:\Program Files\ANSYS Inc\v261\Lumerical\api\python\lumapi.py" 
# spec = importlib.util.spec_from_file_location("lumapi", lumapi_path)
# lumapi = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(lumapi)

# --- 2. Define Simulation Parameters & Toggles ---
wavelengths = np.arange(555e-9, 580e-9, 1e-9)
#wavelengths = np.arange(555e-9, 557e-9, 1e-9)
R = 120e-6 # center radius of the ring     
#G = 400e-9 # gap between ring and waveguide (inner top edge to inner top edge)
G = 300e-9     
W = 400e-9 # top width of the waveguide (and ring)
R_wg = R + G + W # center radius of the waveguide
coupling_angle = 60 #deg
R_ring = 120E-6
L_c = 2 * np.pi * R_ring * (coupling_angle / 360) # coupling length based on angle
#L_c = 126e-6     
c0 = 299792458  


# Diagnostic Plotting Setup
SAVE_DIAGNOSTIC_PLOTS = True
PLOT_DIR = "cmt_diagnostics"
if SAVE_DIAGNOSTIC_PLOTS and not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)

# File paths
DATA_CACHE_FILE = "555_to_579_run.pkl"
FORCE_RERUN = False  # Set to True if you change Lumerical geometry and need fresh data


# --- 2. Load or Extract Data ---
if os.path.exists(DATA_CACHE_FILE) and not FORCE_RERUN:
    print(f"Loading cached simulation data from {DATA_CACHE_FILE}...")
    with open(DATA_CACHE_FILE, 'rb') as f:
        saved_data = pickle.load(f)
        ring_data = saved_data['ring_data']
        wg_data = saved_data['wg_data']
        # Optionally verify wavelengths match
        if not np.array_equal(wavelengths, saved_data['wavelengths']):
            print("WARNING: Script wavelengths do not match cached wavelengths!")
    print("Data loaded successfully. Proceeding to CMT calculations...")

else:
    print("No cache found (or FORCE_RERUN=True). Booting Lumerical...")
    
    # Initialize Lumerical API inside the execution block so it doesn't boot if we are loading from cache
    lumapi_path = "C:\\Program Files\\ANSYS Inc\\v261\\Lumerical\\api\\python\\lumapi.py" 
    spec = importlib.util.spec_from_file_location("lumapi", lumapi_path)
    lumapi = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lumapi)



    ring_data = {'E': [], 'eps': [], 'neff': [], 'ng': [], 'x': None, 'y': None}
    wg_data = {'E': [], 'eps': [], 'neff': [], 'x': None, 'y': None}

    # --- 3. Run Lumerical and Extract Data ---
    def run_and_extract(filename, data_dict):
        with lumapi.MODE(filename, hide=True) as mode:       
            for wl in wavelengths:
                mode.switchtolayout()
                mode.setnamed("FDE", "wavelength", wl)
                mode.run()
                mode.findmodes()
                
                neff = mode.getdata("mode1", "neff")[0][0]
                # if "ng" in mode.getdata("mode1", "ng"):
                #     ng = mode.getdata("mode1", "ng")[0][0]
                # else:
                #     ng = 0 # Fallback if ng isn't calculated
                ng = mode.getdata("mode1","ng")
                
                E_dataset = mode.getresult("mode1", "E")
                # 2. Get the refractive index from the FDE simulation region
                # Lumerical handles anisotropy, so it returns index_x, index_y, index_z. 
                # For isotropic materials like Si3N4/SiO2, they are identical.
                n_y = mode.getdata("material", "index_y")
                
                # Calculate relative permittivity (eps = n^2)
                # We take the real part to drop any tiny imaginary numerical artifacts
                eps = np.real(np.squeeze(n_y))**2
                
                E = np.squeeze(E_dataset['E'])
                #eps = np.squeeze(eps_dataset['eps'])
                x = np.squeeze(E_dataset['y'])
                y = np.squeeze(E_dataset['z'])
                
                data_dict['E'].append(E)
                data_dict['eps'].append(eps)
                data_dict['neff'].append(neff)
                if 'ng' in data_dict:
                    data_dict['ng'].append(ng)
                
                if data_dict['x'] is None:
                    data_dict['x'] = x
                    data_dict['y'] = y

    print("Simulating Ring...")
    run_and_extract("ring_isolated.lms", ring_data) # Uncomment to run

    print("Simulating Waveguide...")
    run_and_extract("waveguide_isolated.lms", wg_data) # Uncomment to run

    # Save the data for future runs
    print(f"Saving extracted data to {DATA_CACHE_FILE}...")
    with open(DATA_CACHE_FILE, 'wb') as f:
        pickle.dump({
            'ring_data': ring_data,
            'wg_data': wg_data,
            'wavelengths': wavelengths
        }, f)
    print("Data saved.")

# Mock data generation for the sake of script completion without Lumerical
# Remove this block when running with actual Lumerical data
# if len(ring_data['E']) == 0:
#     print("Mocking up data for demonstration...")
#     mock_x, mock_y = np.linspace(-1e-6, 1e-6, 100), np.linspace(-1e-6, 1e-6, 100)
#     for w in wavelengths:
#         ring_data['E'].append(np.ones((100, 100, 3), dtype=complex))
#         ring_data['eps'].append(np.ones((100, 100)))
#         ring_data['neff'].append(2.0)
#         ring_data['ng'].append(2.1)
#         wg_data['E'].append(np.ones((100, 100, 3), dtype=complex))
#         wg_data['eps'].append(np.ones((100, 100)))
#         wg_data['neff'].append(1.9)
#     ring_data['x'], ring_data['y'] = mock_x, mock_y
#     wg_data['x'], wg_data['y'] = mock_x, mock_y

# --- 4. Process Data & Generate Plots ---
Qc_results = []

for i, wl in enumerate(wavelengths):
    omega = 2 * np.pi * c0 / wl
    
    E_R = ring_data['E'][i]
    eps_R = ring_data['eps'][i]
    neff_R = np.real(ring_data['neff'][i])
    ng_R = np.squeeze(np.real(ring_data['ng'][i]))
    print('ng_R is',ng_R)
    
    E_wg = wg_data['E'][i]
    eps_wg = wg_data['eps'][i]
    neff_wg = np.real(wg_data['neff'][i])
    
    # Grid Alignment
    offset_x = R_wg - R 
    wg_x_shifted = wg_data['x'] + offset_x
    
    common_x = np.linspace(min(ring_data['x'][0], wg_x_shifted[0]), 
                           max(ring_data['x'][-1], wg_x_shifted[-1]), 500)
    common_y = np.linspace(min(ring_data['y'][0], wg_data['y'][0]), 
                           max(ring_data['y'][-1], wg_data['y'][-1]), 500)
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
    
    # --- DIAGNOSTIC PLOTTING ---
    if SAVE_DIAGNOSTIC_PLOTS:
        # Calculate field magnitudes for visualization
        E_R_mag = np.sqrt(np.sum(np.abs(E_R)**2, axis=-1))
        E_wg_mag = np.sqrt(np.sum(np.abs(E_wg)**2, axis=-1))
        E_R_grid_mag = np.sqrt(np.sum(np.abs(E_R_grid)**2, axis=-1))
        E_wg_grid_mag = np.sqrt(np.sum(np.abs(E_wg_grid)**2, axis=-1))
        
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"CMT Alignment Diagnostics: Wavelength = {wl*1e9:.1f} nm", fontsize=16)
        
        # Plot 1: Isolated Ring (Original Grid)
        axs[0, 0].pcolormesh(ring_data['x']*1e6, ring_data['y']*1e6, E_R_mag.T, shading='auto', cmap='magma')
        #axs[0, 0].contour(ring_data['x']*1e6, ring_data['y']*1e6, eps_R.T, levels=[2.0], colors='white', linestyles='dashed', alpha=0.7)
        axs[0, 0].contour(ring_data['x']*1e6, ring_data['y']*1e6, eps_R.T, 
                  levels=[(np.max(eps_R) + np.min(eps_R))/2], 
                  colors='white', linestyles='dashed', alpha=0.7)
        axs[0, 0].set_title("1. Isolated Ring (Mode & Index)")
        axs[0, 0].set_xlabel("x (um)"); axs[0, 0].set_ylabel("y (um)")
        
        # Plot 2: Isolated Waveguide (Original Grid)
        axs[0, 1].pcolormesh(wg_data['x']*1e6, wg_data['y']*1e6, E_wg_mag.T, shading='auto', cmap='magma')
        #axs[0, 1].contour(wg_data['x']*1e6, wg_data['y']*1e6, eps_wg.T, levels=[2.0], colors='white', linestyles='dashed', alpha=0.7)
        axs[0, 1].contour(wg_data['x']*1e6, wg_data['y']*1e6, eps_wg.T, 
                  levels=[(np.max(eps_wg) + np.min(eps_wg))/2], 
                  colors='white', linestyles='dashed', alpha=0.7)
        axs[0, 1].set_title("2. Isolated Waveguide (Mode & Index)")
        axs[0, 1].set_xlabel("x (um)"); axs[0, 1].set_ylabel("y (um)")
        
        # Plot 3: Combined Modes on Common Grid
        axs[1, 0].pcolormesh(common_x*1e6, common_y*1e6, (E_R_grid_mag + E_wg_grid_mag).T, shading='auto', cmap='magma')
        axs[1, 0].contour(common_x*1e6, common_y*1e6, eps_R_grid.T, levels=[2.0], colors='cyan', alpha=0.8)
        axs[1, 0].contour(common_x*1e6, common_y*1e6, eps_wg_grid.T, levels=[2.0], colors='lime', alpha=0.8)
        axs[1, 0].set_title("3. Aligned Modes on Common Grid")
        axs[1, 0].set_xlabel("x (um)"); axs[1, 0].set_ylabel("y (um)")
        
        # Plot 4: Index Mask (eps_wg - eps_R) on Common Grid
        mask = eps_wg_grid - eps_R_grid
        im = axs[1, 1].pcolormesh(common_x*1e6, common_y*1e6, mask.T, shading='auto', cmap='RdBu', vmin=-max(np.abs(mask.flatten())), vmax=max(np.abs(mask.flatten())))
        axs[1, 1].set_title("4. CMT Spatial Mask (eps_wg - eps_R)")
        axs[1, 1].set_xlabel("x (um)"); axs[1, 1].set_ylabel("y (um)")
        fig.colorbar(im, ax=axs[1, 1], label="Permittivity Difference")
        
        plt.tight_layout()
        plot_path = os.path.join(PLOT_DIR, f"diagnostic_wl_{wl*1e9:.1f}nm.png")
        plt.savefig(plot_path, dpi=150)
        plt.close(fig) # Closes the figure so it doesn't show in interactive environments

    # CMT Calculations
    delta_beta = (omega / c0) * (neff_R * (R / R_wg) - neff_wg)
    
    # dot_product = np.sum(np.conj(E_R_grid) * E_wg_grid, axis=2) 
    # integrand = (eps_wg_grid - eps_R_grid) * dot_product
    
    dx = common_x[1] - common_x[0]
    dy = common_y[1] - common_y[0]

    eta0 = 376.730313668  # Free space impedance in ohms
    # Calculate current power of the raw grids
    P_R = (neff_R / (2 * eta0)) * np.sum(np.abs(E_R_grid)**2) * dx * dy
    P_wg = (neff_wg / (2 * eta0)) * np.sum(np.abs(E_wg_grid)**2) * dx * dy
    print('P_R is',P_R)
    print('P_wg is',P_wg)

    # Normalize fields
    E_R_grid_norm = E_R_grid / np.sqrt(P_R)
    E_wg_grid_norm = E_wg_grid / np.sqrt(P_wg)

    eps0 = 8.854187817e-12  # Vacuum permittivity in F/m

    dot_product = np.sum(np.conj(E_R_grid_norm) * E_wg_grid_norm, axis=2) 
    integrand = eps0*(eps_wg_grid - eps_R_grid) * dot_product

    Gamma = (1j * omega / 4) * np.sum(integrand) * dx * dy
    Gamma_mag = np.abs(Gamma)
    
    sinc_arg = L_c * np.sqrt((delta_beta/2)**2 + Gamma_mag**2)
    sinc_val = np.sin(sinc_arg) / sinc_arg 
    
    coupling_factor = (Gamma_mag * L_c * sinc_val)**(-2)
    Qc = omega * (ng_R / c0) * 2 * np.pi * R * coupling_factor

    print('ng_R type is',type(ng_R))
    
    # --- INTERMEDIATE DIAGNOSTIC SUMMARY ---
    print(f"\n--- Intermediate CMT Math for {wl*1e9:.1f} nm ---")
    print(f"neff_R:          {neff_R:.4f}")
    print(f"neff_wg:         {neff_wg:.4f}")
    print(f"ng_R:            {ng_R:.4f}  <-- WARNING: If 0, Qc will be 0!")
    print(f"delta_beta:      {delta_beta:.2e} rad/m")
    print(f"Gamma (complex): {Gamma.real:.2e} + {Gamma.imag:.2e}j")
    print(f"Gamma_mag:       {Gamma_mag:.2e}")
    print(f"sinc_arg:        {sinc_arg:.4f}")
    print(f"sinc_val:        {sinc_val:.4f}")
    print(f"coupling_factor: {coupling_factor:.2e}")
    print(f"Final Qc:        {Qc:.2e}")
    print("------------------------------------------")

    # --- INTEGRAND VISUALIZATION ---
    if SAVE_DIAGNOSTIC_PLOTS:
        fig2, axs2 = plt.subplots(1, 2, figsize=(12, 5))
        fig2.suptitle(f"Integrand Visualization: Wavelength = {wl*1e9:.1f} nm", fontsize=14)
        
        # Plot Real part of the integrand
        vmax_real = np.max(np.abs(np.real(integrand)))
        im0 = axs2[0].pcolormesh(common_x*1e6, common_y*1e6, np.real(integrand).T, 
                                 cmap='seismic', shading='auto', vmin=-vmax_real, vmax=vmax_real)
        axs2[0].set_title("Real Part of Integrand")
        axs2[0].set_xlabel("x (um)"); axs2[0].set_ylabel("y (um)")
        fig2.colorbar(im0, ax=axs2[0])
        
        # Plot Imaginary part of the integrand
        vmax_imag = np.max(np.abs(np.imag(integrand)))
        im1 = axs2[1].pcolormesh(common_x*1e6, common_y*1e6, np.imag(integrand).T, 
                                 cmap='seismic', shading='auto', vmin=-vmax_imag, vmax=vmax_imag)
        axs2[1].set_title("Imaginary Part of Integrand")
        axs2[1].set_xlabel("x (um)"); axs2[1].set_ylabel("y (um)")
        fig2.colorbar(im1, ax=axs2[1])
        
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f"integrand_wl_{wl*1e9:.1f}nm.png"), dpi=150)
        plt.close(fig2)

    Qc_results.append(Qc)

for wl, q in zip(wavelengths, Qc_results):
    print(f"Wavelength: {wl*1e9:.0f} nm -> Qc: {q:.2e}")

# --- 5. Plot Qc Sweep ---
# Optional code-level override. CLI flag --qc-plot-name takes precedence.
QC_PLOT_FILENAME = None

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
default_qc_plot_filename = f"qc_sweep_{timestamp}.png"

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--qc-plot-name", type=str, default=None)
cli_args, _ = parser.parse_known_args()

qc_plot_filename = cli_args.qc_plot_name or QC_PLOT_FILENAME or default_qc_plot_filename

wl_nm = wavelengths * 1e9
qc_vals = np.array(Qc_results)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(wl_nm, qc_vals, linestyle=':', linewidth=1.2, color='purple', alpha=0.5)
ax.scatter(wl_nm, qc_vals, color='purple', s=28)
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Qc")
ax.set_title("Qc vs Wavelength")

plt.savefig(qc_plot_filename, dpi=500, bbox_inches='tight')
plt.close(fig)
print(f"Saved Qc sweep plot to {qc_plot_filename}")