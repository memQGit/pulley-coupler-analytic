import numpy as np
import importlib.util
import os
import pickle

# --- 1. Define Simulation Parameters ---
wavelengths = np.array([556e-9])
R = 120e-6     # center radius of the ring     
G = 175e-9     # gap between ring and waveguide
W_r = 400e-9     # top width of the ring waveguide
W_wg = 400e-9     # top width of the waveguide
wg_height = 100e-9 # height of waveguide
c0 = 299792458 
sidewall = 65 # sidewall angle in degrees. 90 is vertical.

DATA_CACHE_FILE = f"{W_r*1e9:.0f}nm_ring_{W_wg*1e9:.0f}nm_wg_{wavelengths[0]*1e9:.0f}nm_light_{G*1e9:.0f}nm_gap.pkl"

# --- 2. Initialize Lumerical API ---
print("Booting Lumerical...")
lumapi_path = "C:\\Program Files\\ANSYS Inc\\v261\\Lumerical\\api\\python\\lumapi.py" 
spec = importlib.util.spec_from_file_location("lumapi", lumapi_path)
lumapi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lumapi)

ring_data = {'E': [], 'eps': [], 'neff': [], 'ng': [], 'x': None, 'y': None}
wg_data = {'E': [], 'eps': [], 'neff': [], 'x': None, 'y': None}
params = {'R': R, 'G': G, 'W_r': W_r, 'W_wg': W_wg, 'sidewall' : sidewall, 'c0': c0, 'wavelengths': wavelengths, 'wg_height': wg_height}

# --- 3. Run Lumerical and Extract Data ---
def run_and_extract(filename, data_dict, param_dict, width_key='W_r'):
    with lumapi.MODE(filename, hide=False) as mode:       
        for wl in param_dict['wavelengths']:
            mode.switchtolayout()
            mode.setnamed("FDE", "wavelength", wl)
            mode.setnamed("LN1", "base width", param_dict[width_key]+(2*param_dict['wg_height']/np.tan(np.radians(param_dict['sidewall']))))
            mode.setnamed("LN1", "base angle", param_dict['sidewall'])    
            mode.setnamed("FDE", "bent waveguide", True)
            mode.setnamed("FDE", "bend radius", param_dict['R'])
            
            mode.run()
            mode.findmodes()
            
            neff = mode.getdata("mode1", "neff")[0][0]
            ng = mode.getdata("mode1","ng")
            
            E_dataset = mode.getresult("mode1", "E")
            n_y = mode.getdata("material", "index_y")
            eps = np.real(np.squeeze(n_y))**2
            
            E = np.squeeze(E_dataset['E'])
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
run_and_extract("ring_isolated.lms", ring_data, params, width_key='W_r')

print("Simulating Waveguide...")
run_and_extract("waveguide_isolated.lms", wg_data, params, width_key='W_wg')

# Save the data AND physical parameters for future runs
print(f"Saving extracted data to {DATA_CACHE_FILE}...")

with open(DATA_CACHE_FILE, 'wb') as f:
    pickle.dump({
        'ring_data': ring_data,
        'wg_data': wg_data,
        'wavelengths': wavelengths,
        'params': params
    }, f)
print("Data saved successfully.")