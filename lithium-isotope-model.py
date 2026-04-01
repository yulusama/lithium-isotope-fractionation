
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import pandas as pd

width_cm = 34.0
height_cm = 18.0
plt.rcParams['figure.figsize'] = (width_cm / 2.54, height_cm / 2.54)
plt.rcParams['font.family'] = 'Calibri'
plt.rcParams['font.size'] = 15
plt.rcParams['axes.labelsize'] = 15
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams['lines.linewidth'] = 1.2
plt.rcParams['axes.unicode_minus'] = False

data = [
    [244.00, 14.8, 0.92, 0.038],
    [244.15, 9.6, 1.47, 0.063],
    [245.18, 7.3, 0.12, 0.01],
    [245.69, 7.8, 0.23, 0.012],
    [246.03, 7.8, 0.15, 0.008],
    [246.46, 7.8, 0.28, 0.016],
    [246.87, 6.0, 0.49, 0.036],
    [247.20, 3.0, 1.18, 0.106],
    [247.70, 3.3, 0.48, 0.047],
    [248.35, 8.1, 0.28, 0.016],
    [249.20, 5.4, 0.51, 0.037],
    [249.34, 8.1, 0.13, 0.007],
    [249.46, 18.4, 0.32, 0.02],
    [249.53, 15.7, 0.41, 0.019],
    [250.11, 15.8, 0.52, 0.04],
    [250.21, 6.2, 0.35, 0.048],
    [250.60, 8.0, 3.13, 0.352],
    [251.08, 3.4, 1.68, 0.155],
    [251.52, 6.0, 1.13, 0.104]
]
df_data = pd.DataFrame(data, columns=['Age', 'dLi_orig', 'Al', 'Ti'])

dLi_ref = 8.1
CO2_ref = 2400
t_peak = 249.34

Al_det_mean = 8.0
dLi_det_mean = 1.0
Delta_carb_sw_mean = 4.0

def correct_dLi(row, Al_det, dLi_det, Delta_carb_sw):
    f_det = row['Al'] / Al_det
    if f_det >= 1.0:
        f_det = 0.99
    dLi_carb = (row['dLi_orig'] - f_det * dLi_det) / (1 - f_det)
    dLi_sw = dLi_carb + Delta_carb_sw
    return dLi_sw

df_data['dLi_corr'] = df_data.apply(lambda row: correct_dLi(row, Al_det_mean, dLi_det_mean, Delta_carb_sw_mean), axis=1)

ages_orig = df_data['Age'].values
dLi_orig_vals = df_data['dLi_orig'].values
dLi_corr_vals = df_data['dLi_corr'].values

F0 = 0.0002
A_LIP = 0.07
sigma_LIP = 1.0
t_LIP = 249.4
A_OAB = 0.01
sigma_OAB = 0.15
t_OAB = 247.2

eta_pos = 0.15
eta_rev0 = 0.10
k_T = 0.5
Fs0 = 1.2
E_sil = 0.8

def alpha_smooth(t):
    alpha_early = 0.05
    alpha_late = 0.16
    t_start = 249.0
    t_end = 248.0
    if t >= t_start:
        return alpha_early
    elif t <= t_end:
        return alpha_late
    else:
        return alpha_early + (alpha_late - alpha_early) * (t_start - t) / (t_start - t_end)

beta_osc = 0.15
T_osc = 3.0
t0_osc = 252.0
lambda_T = 4.0
window_width = 1.0

def volcanic_flux_base(t):
    FLIP = A_LIP * np.exp(-(t - t_LIP) ** 2 / (2 * sigma_LIP ** 2))
    FOAB = A_OAB * np.exp(-(t - t_OAB) ** 2 / (2 * sigma_OAB ** 2))
    return F0 + FLIP + FOAB

def volcanic_flux_theory(t):
    if t > 249.34:
        FLIP = A_LIP * np.exp(-(t - t_LIP) ** 2 / (2 * sigma_LIP ** 2))
        return F0 + FLIP
    else:
        FLIP_at_peak = A_LIP * np.exp(-(249.34 - t_LIP) ** 2 / (2 * sigma_LIP ** 2))
        base_at_peak = F0 + FLIP_at_peak
        tau = 0.3
        decay = np.exp(-(249.34 - t) / tau)
        return F0 + (base_at_peak - F0) * decay

ages_extra = np.linspace(242.0, 244.0, 500, endpoint=True)
ages_meas = np.linspace(244.0, 252.0, 1500, endpoint=False)
ages_full = np.concatenate([ages_extra, ages_meas])

Li_eq_typical = 11.0
tau_typical = 1.0

def extrapolate_dLi(t, t_peak, Li_peak, Li_eq, tau):
    return Li_eq + (Li_peak - Li_eq) * np.exp(-(t_peak - t) / tau)

Li_peak_typical = dLi_corr_vals[0]

dLi_full = np.zeros_like(ages_full)
for i, t in enumerate(ages_full):
    if t > 244.0:
        dLi_full[i] = np.interp(t, ages_orig, dLi_corr_vals)
    elif t < 244.0:
        dLi_full[i] = extrapolate_dLi(t, 244.0, Li_peak_typical, Li_eq_typical, tau_typical)
    else:
        dLi_full[i] = Li_peak_typical

def solve_series(ages, dLi, volcanic_func, params_dict):
    ages_desc = ages[::-1]
    dLi_desc = dLi[::-1]

    results = []
    prev_dT = None
    for t, dLi_val in zip(ages_desc, dLi_desc):
        x = np.log(dLi_val / dLi_ref)
        Fvol = volcanic_func(t)

        beta = params_dict['beta_osc']
        T = params_dict['T_osc']
        t_start = 244.0
        t_end = 252.0
        width = window_width

        if t < t_start - width:
            w = 0.0
        elif t < t_start:
            x = (t - (t_start - width)) / width
            w = 3.0 * x ** 2 - 2.0 * x ** 3
        elif t <= t_end:
            w = 1.0
        elif t < t_end + width:
            x = (t_end + width - t) / width
            w = 3.0 * x ** 2 - 2.0 * x ** 3
        else:
            w = 0.0

        osc = 1 + beta * np.sin(2 * np.pi * (t - t0_osc) / T) * w

        alpha_use = alpha_smooth(t)

        dT = prev_dT if prev_dT is not None else 0.0
        tol = 1e-4
        for _ in range(30):
            eta_rev = params_dict['eta_rev0'] * (1 + params_dict['k_T'] * max(0, dT))
            eta_eff = params_dict['eta_pos'] - eta_rev
            Fsil = params_dict['Fs0'] * np.exp(params_dict['E_sil'] * dT)
            Fclay = np.exp(eta_eff * x)
            Fw = Fsil + Fclay
            r = Fvol / Fw
            CO2 = CO2_ref * (r ** alpha_use) * osc
            dT_new = params_dict['lambda_T'] * np.log(CO2 / CO2_ref)
            if abs(dT_new - dT) < tol:
                dT = dT_new
                eta_rev = params_dict['eta_rev0'] * (1 + params_dict['k_T'] * max(0, dT))
                eta_eff = params_dict['eta_pos'] - eta_rev
                Fsil = params_dict['Fs0'] * np.exp(params_dict['E_sil'] * dT)
                Fclay = np.exp(eta_eff * x)
                Fw = Fsil + Fclay
                r = Fvol / Fw
                CO2 = CO2_ref * (r ** alpha_use) * osc
                break
            dT = dT_new

        results.append({
            'Age': t,
            'CO2': CO2,
            'dT': dT,
            'r': r,
            'Fvol': Fvol,
            'Fsil': Fsil,
            'Fclay': Fclay
        })
        prev_dT = dT

    df_res = pd.DataFrame(results).sort_values('Age').reset_index(drop=True)
    return {key: df_res[key].values for key in ['Age', 'CO2', 'dT', 'r', 'Fvol', 'Fsil', 'Fclay']}

params_typical = {
    'eta_pos': eta_pos,
    'eta_rev0': eta_rev0,
    'k_T': k_T,
    'Fs0': Fs0,
    'E_sil': E_sil,
    'beta_osc': beta_osc,
    'T_osc': T_osc,
    'lambda_T': lambda_T
}

res_base = solve_series(ages_full, dLi_full, volcanic_flux_base, params_typical)
ages_out = res_base['Age']
CO2_base = res_base['CO2']
dT_base = res_base['dT']
r_base = res_base['r']
Fvol_base = res_base['Fvol']
Fsil_base = res_base['Fsil']
Fclay_base = res_base['Fclay']

res_th = solve_series(ages_full, dLi_full, volcanic_flux_theory, params_typical)
CO2_th = res_th['CO2']
dT_th = res_th['dT']

dT_diff = dT_base - dT_th

param_ranges = {
    'Al_det': (7.0, 9.0),
    'dLi_det': (0.0, 2.0),
    'Delta_carb_sw': (3.0, 6.0),
    'F0': (0.0001, 0.0003),
    'A_LIP': (0.06, 0.08),
    'sigma_LIP': (0.9, 1.1),
    'A_OAB': (0.008, 0.012),
    'sigma_OAB': (0.12, 0.18),
    'eta_pos': (0.13, 0.17),
    'eta_rev0': (0.08, 0.12),
    'k_T': (0.4, 0.6),
    'Fs0': (1.0, 1.4),
    'E_sil': (0.7, 0.9),
    'alpha_early': (0.04, 0.06),
    'alpha_late': (0.14, 0.18),
    'beta_osc': (0.12, 0.18),
    'T_osc': (2.5, 5.0),
    'lambda_T': (3.5, 4.5),
    'Li_eq': (10.0, 12.0),
    'tau': (0.5, 1.5)
}

N_MC = 500
np.random.seed(42)
n_times = len(ages_full)
CO2_samples = np.zeros((N_MC, n_times))
dT_samples = np.zeros((N_MC, n_times))

Al_vals = df_data['Al'].values.tolist()
dLi_orig_list = df_data['dLi_orig'].values.tolist()

def get_corrected_dLi_full(params_i, ages_full):
    dLi_corr_orig = []
    for i, dLi_orig in enumerate(dLi_orig_list):
        f_det = Al_vals[i] / params_i['Al_det']
        if f_det >= 1.0:
            f_det = 0.99
        dLi_carb = (dLi_orig - f_det * params_i['dLi_det']) / (1 - f_det)
        dLi_sw = dLi_carb + params_i['Delta_carb_sw']
        dLi_corr_orig.append(dLi_sw)

    Li_peak = dLi_corr_orig[0]
    dLi_full = np.zeros_like(ages_full)
    for j, t in enumerate(ages_full):
        if t > 244.0:
            dLi_full[j] = np.interp(t, ages_orig, dLi_corr_orig)
        elif t < 244.0:
            dLi_full[j] = params_i['Li_eq'] + (Li_peak - params_i['Li_eq']) * np.exp(-(244.0 - t) / params_i['tau'])
        else:
            dLi_full[j] = Li_peak
    return dLi_full

print("Running Monte Carlo...")
for i in range(N_MC):
    if (i + 1) % 100 == 0:
        print(f"  {i + 1}/{N_MC}")

    params_i = {key: np.random.uniform(low, high) for key, (low, high) in param_ranges.items()}
    dLi_full_i = get_corrected_dLi_full(params_i, ages_full)

    solver_params = {
        'eta_pos': params_i['eta_pos'],
        'eta_rev0': params_i['eta_rev0'],
        'k_T': params_i['k_T'],
        'Fs0': params_i['Fs0'],
        'E_sil': params_i['E_sil'],
        'beta_osc': params_i['beta_osc'],
        'T_osc': params_i['T_osc'],
        'lambda_T': params_i['lambda_T']
    }

    def vf_base_i(t):
        FLIP = params_i['A_LIP'] * np.exp(-(t - t_LIP) ** 2 / (2 * params_i['sigma_LIP'] ** 2))
        FOAB = params_i['A_OAB'] * np.exp(-(t - t_OAB) ** 2 / (2 * params_i['sigma_OAB'] ** 2))
        return params_i['F0'] + FLIP + FOAB

    res_i = solve_series(ages_full, dLi_full_i, vf_base_i, solver_params)
    CO2_samples[i, :] = res_i['CO2']
    dT_samples[i, :] = res_i['dT']

percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
CO2_perc = np.percentile(CO2_samples, percentiles, axis=0).T
dT_perc = np.percentile(dT_samples, percentiles, axis=0).T

fig = plt.figure(figsize=(width_cm / 2.54, height_cm / 2.54))
gs = fig.add_gridspec(3, 3, hspace=0.25, wspace=0.25, left=0.08, right=0.98, top=0.98, bottom=0.08)

color_orig = '#212121'
color_corr = '#C1272D'
color_volcano = '#E67E22'
color_co2 = '#5B2C90'
color_temp = '#255499'
color_fsil = '#238B84'
color_fclay = '#884EA0'
color_r = '#333333'

def plot_age_split(ax, x, y, solid_color, dashed_color=None, label_solid='', label_dashed='', **kwargs):
    if dashed_color is None:
        dashed_color = solid_color
    idx_split = np.searchsorted(x, 244.0)
    ax.plot(x[:idx_split], y[:idx_split], '--', color=dashed_color, label=label_dashed, **kwargs)
    ax.plot(x[idx_split:], y[idx_split:], '-', color=solid_color, label=label_solid, **kwargs)

ax1 = fig.add_subplot(gs[0, 0])
p_orig = np.polyfit(ages_orig, dLi_orig_vals, 3)
p_corr = np.polyfit(ages_orig, dLi_corr_vals, 3)
ages_fit = np.linspace(ages_orig.min(), ages_orig.max(), 200)
ax1.plot(ages_fit, np.polyval(p_orig, ages_fit), color=color_orig, label='Bulk rock')
ax1.plot(ages_fit, np.polyval(p_corr, ages_fit), color=color_corr, label='Seawater (corrected)')
ax1.scatter(ages_orig, dLi_orig_vals, color=color_orig, s=20, marker='o', facecolors='none', edgecolors=color_orig, linewidth=0.8)
ax1.scatter(ages_orig, dLi_corr_vals, color=color_corr, s=20, marker='s', facecolors='none', edgecolors=color_corr, linewidth=0.8)
ax1.set_ylabel('δ⁷Li (‰)')
ax1.legend(loc='upper right', frameon=False)
ax1.grid(True, alpha=0.3)
ax1.invert_xaxis()

ax2 = fig.add_subplot(gs[0, 1])
plot_age_split(ax2, ages_full, Fvol_base, solid_color=color_volcano, label_solid='Measured', label_dashed='Extrapolated')
ax2.set_ylabel('Volcanic flux')
ax2.grid(True, alpha=0.3)
ax2.invert_xaxis()

ax3 = fig.add_subplot(gs[0, 2])
plot_age_split(ax3, ages_full, CO2_base, solid_color=color_co2, label_solid='Measured', label_dashed='Extrapolated')
ax3.axhline(CO2_ref, color='gray', linestyle='--', alpha=0.5)
ax3.set_ylabel('CO₂ (ppm)')
ax3.grid(True, alpha=0.3)
ax3.invert_xaxis()

ax4 = fig.add_subplot(gs[1, 0])
plot_age_split(ax4, ages_full, dT_base, solid_color=color_temp, label_solid='Baseline (meas)', label_dashed='Baseline (ext)')
plot_age_split(ax4, ages_full, dT_th, solid_color=color_temp, dashed_color=color_temp, label_solid='Theoretical (meas)', label_dashed='Theoretical (ext)')
ax4.set_ylabel('ΔT (°C)')
ax4.legend(loc='upper right', frameon=False)
ax4.grid(True, alpha=0.3)
ax4.invert_xaxis()

ax5 = fig.add_subplot(gs[1, 1])
plot_age_split(ax5, ages_full, dT_diff, solid_color=color_volcano, label_solid='Volcanic contr. (meas)', label_dashed='Volcanic contr. (ext)')
ax5.fill_between(ages_full, 0, dT_diff, color=color_volcano, alpha=0.7)
ax5.set_ylabel('ΔT_diff (°C)')
ax5.legend(loc='upper left', frameon=False)
ax5.grid(True, alpha=0.3)
ax5.invert_xaxis()

ax6 = fig.add_subplot(gs[1, 2])
ax6.fill_between(ages_full, CO2_perc[:, 0], CO2_perc[:, 8], color=color_co2, alpha=0.1, label='1–99%')
ax6.fill_between(ages_full, CO2_perc[:, 1], CO2_perc[:, 7], color=color_co2, alpha=0.2, label='5–95%')
ax6.fill_between(ages_full, CO2_perc[:, 2], CO2_perc[:, 6], color=color_co2, alpha=0.35, label='10–90%')
plot_age_split(ax6, ages_full, CO2_perc[:, 4], solid_color=color_co2, label_solid='Median (meas)', label_dashed='Median (ext)')
ax6.set_ylabel('CO₂ (ppm)')
ax6.legend(loc='upper right', frameon=False)
ax6.grid(True, alpha=0.3)
ax6.invert_xaxis()
ax6.axvline(244.0, color='gray', linestyle=':', linewidth=1)

ax7 = fig.add_subplot(gs[2, 0])
ax7.fill_between(ages_full, dT_perc[:, 0], dT_perc[:, 8], color=color_temp, alpha=0.1, label='1–99%')
ax7.fill_between(ages_full, dT_perc[:, 1], dT_perc[:, 7], color=color_temp, alpha=0.2, label='5–95%')
ax7.fill_between(ages_full, dT_perc[:, 2], dT_perc[:, 6], color=color_temp, alpha=0.35, label='10–90%')
plot_age_split(ax7, ages_full, dT_perc[:, 4], solid_color=color_temp, label_solid='Median (meas)', label_dashed='Median (ext)')
ax7.set_xlabel('Age (Ma)')
ax7.set_ylabel('ΔT (°C)')
ax7.legend(loc='lower left', frameon=False)
ax7.grid(True, alpha=0.3)
ax7.invert_xaxis()
ax7.axvline(244.0, color='gray', linestyle=':', linewidth=1)

ax8 = fig.add_subplot(gs[2, 1])
plot_age_split(ax8, ages_full, Fsil_base, solid_color=color_fsil, label_solid='Fsil (meas)', label_dashed='Fsil (ext)')
plot_age_split(ax8, ages_full, Fclay_base, solid_color=color_fclay, label_solid='Fclay (meas)', label_dashed='Fclay (ext)')
ax8.set_xlabel('Age (Ma)')
ax8.set_ylabel('Weathering flux')
ax8.legend(loc='upper right', frameon=False)
ax8.grid(True, alpha=0.3)
ax8.invert_xaxis()
ax8.axvline(244.0, color='gray', linestyle=':', linewidth=1)

ax9 = fig.add_subplot(gs[2, 2])
plot_age_split(ax9, ages_full, r_base, solid_color=color_r, label_solid='r (meas)', label_dashed='r (ext)')
ax9.set_xlabel('Age (Ma)')
ax9.set_ylabel('r = Fvol/Fw')
ax9.grid(True, alpha=0.3)
ax9.invert_xaxis()
ax9.axvline(244.0, color='gray', linestyle=':', linewidth=1)

plt.savefig('Nature_style_panel_final.eps', format='eps', dpi=300, bbox_inches='tight')
plt.show()

with pd.ExcelWriter('reconstruction_data_final.xlsx', engine='openpyxl') as writer:
    df_orig = pd.DataFrame({'Age': ages_orig, 'dLi_orig': dLi_orig_vals, 'Al': df_data['Al'], 'Ti': df_data['Ti'], 'dLi_corr': dLi_corr_vals})
    df_orig.to_excel(writer, sheet_name='Raw_corrected', index=False)

    df_base = pd.DataFrame({'Age': ages_full, 'CO2_base': CO2_base, 'dT_base': dT_base, 'r_base': r_base,
                            'Fvol_base': Fvol_base, 'Fsil_base': Fsil_base, 'Fclay_base': Fclay_base})
    df_base.to_excel(writer, sheet_name='Baseline', index=False)

    df_th = pd.DataFrame({'Age': ages_full, 'CO2_th': CO2_th, 'dT_th': dT_th})
    df_th.to_excel(writer, sheet_name='Theoretical', index=False)

    df_diff = pd.DataFrame({'Age': ages_full, 'dT_base': dT_base, 'dT_th': dT_th, 'dT_diff': dT_diff})
    df_diff.to_excel(writer, sheet_name='Temperature_diff', index=False)

    df_CI = pd.DataFrame(np.column_stack([ages_full, CO2_perc]), columns=['Age'] + [f'CO2_p{p}' for p in percentiles])
    df_CI.to_excel(writer, sheet_name='CO2_uncertainty', index=False)

    df_CI_dT = pd.DataFrame(np.column_stack([ages_full, dT_perc]), columns=['Age'] + [f'dT_p{p}' for p in percentiles])
    df_CI_dT.to_excel(writer, sheet_name='dT_uncertainty', index=False)

print("Figure saved as Nature_style_panel_final.eps")
print("Data saved to reconstruction_data_final.xlsx")

