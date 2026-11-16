import json
import matplotlib.pyplot as plt
import os
import numpy as np

neurons_list = 1, 2, 4, 8, 16, 32, 64, 128, 256, 512
methods = ['FB', 'NW', 'DNW']
labels = {'FB': 'Full-Batched', 'NW': 'Neuron-Wise', 'DNW': 'Double Neuron-Wise'}

methods_n = ['SMC']
labels_n = {'SMC': 'Standard Monte Carlo'}

My_Blue   = (0,        0/255,   102/255)
My_Orange = (255/255, 178/255,  102/255)
My_Green  = (10/255,  102/255,   28/255)

colors   = {'FB': My_Blue, 'NW': My_Orange, 'DNW': My_Green}
markers  = {'FB': 'o',     'NW': 's',       'DNW': '^'}
colors_n = {'SMC': 'red'}
markers_n = {'SMC': 'o'}

cmc_path_200  = 'Deep_Ritz_2D_CMC_time_report/timing_results_CMC_200.json'
smc_path_200  = 'Deep_Ritz_2D_MonteCarlo_time_report/timing_results_Exact_200.json'
cmc_path_500  = 'Deep_Ritz_2D_CMC_time_report/timing_results_CMC_500.json'
smc_path_500  = 'Deep_Ritz_2D_MonteCarlo_time_report/timing_results_Exact_500.json'
cmc_path_1000 = 'Deep_Ritz_2D_CMC_time_report/timing_results_CMC_1000.json'
smc_path_1000 = 'Deep_Ritz_2D_MonteCarlo_time_report/timing_results_Exact_1000.json'

font = {'family': 'Times', 'weight': 'normal', 'size': 13}
plt.rc('font', **font)
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{mathrsfs} \usepackage{amssymb}'

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 3.5))

data_cmc_200 = data_smc_200 = None
data_cmc_500 = data_smc_500 = None
data_cmc_1000 = data_smc_1000 = None

# --- Plot 1: 200 Integration Points ---
ax1.set_xlabel('Number of Neurons', fontsize=15)
ax1.set_ylabel('Time for each Epoch (sec)', fontsize=15)
ax1.set_title('200 Integration Points', fontsize=15)
ax1.grid(True, linestyle='--', alpha=0.7)
if os.path.exists(cmc_path_200):
    with open(cmc_path_200, 'r') as f:
        data_cmc_200 = json.load(f)
    with open(smc_path_200, 'r') as f:
        data_smc_200 = json.load(f)
    for method in methods:
        ax1.plot(neurons_list, data_cmc_200[method], marker=markers[method],
                 color=colors[method], linestyle='-', linewidth=1.5, markersize=6,
                 label=labels[method])
    for method in methods_n:
        ax1.plot(neurons_list, data_smc_200[method], marker=markers_n[method],
                 color=colors_n[method], linestyle='-', linewidth=1.5, markersize=6,
                 label=labels_n[method])
    ax1.set_yscale('log')
    ax1.set_xscale('log')
else:
    print(f"File {cmc_path_200} not found. Please run the experiments first.")

# --- Plot 2: 500 Integration Points ---
ax2.set_xlabel('Number of Neurons', fontsize=15)          # FIX: was ax1
ax2.set_ylabel('Time for each Epoch (sec)', fontsize=15)  # FIX: was ax1
ax2.set_title('500 Integration Points', fontsize=15)      # FIX: was ax1
ax2.grid(True, linestyle='--', alpha=0.7)                 # FIX: was ax1
if os.path.exists(cmc_path_500):
    with open(cmc_path_500, 'r') as f:
        data_cmc_500 = json.load(f)
    with open(smc_path_500, 'r') as f:
        data_smc_500 = json.load(f)
    for method in methods:
        ax2.plot(neurons_list, data_cmc_500[method], marker=markers[method],
                 color=colors[method], linestyle='-', linewidth=1.5, markersize=6,
                 label=labels[method])
    for method in methods_n:
        ax2.plot(neurons_list, data_smc_500[method], marker=markers_n[method],
                 color=colors_n[method], linestyle='-', linewidth=1.5, markersize=6,
                 label=labels_n[method])
    ax2.set_yscale('log')
    ax2.set_xscale('log')
else:
    print(f"File {cmc_path_500} not found. Please run the experiments first.")  # FIX: was cmc_path

# --- Plot 3: 1000 Integration Points ---
ax3.set_xlabel('Number of Neurons', fontsize=15)           # FIX: was ax1
ax3.set_ylabel('Time for each Epoch (sec)', fontsize=15)   # FIX: was ax1
ax3.set_title('1000 Integration Points', fontsize=15)      # FIX: was ax1
ax3.grid(True, linestyle='--', alpha=0.7)                  # FIX: was ax1
if os.path.exists(cmc_path_1000):
    with open(cmc_path_1000, 'r') as f:
        data_cmc_1000 = json.load(f)
    with open(smc_path_1000, 'r') as f:
        data_smc_1000 = json.load(f)
    for method in methods:
        ax3.plot(neurons_list, data_cmc_1000[method], marker=markers[method],
                 color=colors[method], linestyle='-', linewidth=1.5, markersize=6,
                 label=labels[method])
    for method in methods_n:
        ax3.plot(neurons_list, data_smc_1000[method], marker=markers_n[method],
                 color=colors_n[method], linestyle='-', linewidth=1.5, markersize=6,
                 label=labels_n[method])
    ax3.set_yscale('log')
    ax3.set_xscale('log')
else:
    print(f"File {cmc_path_1000} not found. Please run the experiments first.")  # FIX: was cmc_path

# --- Shared y-axis range ---
# FIX: collect from the correctly-named variables (data_cmc_200/500/1000, data_smc_*)
all_vals = []
for data, mlist in [
    (data_cmc_200,  methods),   (data_smc_200,  methods_n),
    (data_cmc_500,  methods),   (data_smc_500,  methods_n),
    (data_cmc_1000, methods),   (data_smc_1000, methods_n),
]:
    if data is not None:
        for m in mlist:
            all_vals.extend([v for v in data[m] if v > 0])

if all_vals:
    y_min_global = min(all_vals)
    y_max_global = max(all_vals)
    for ax in [ax1, ax2, ax3]:
        ax.set_ylim([y_min_global * 0.8, y_max_global * 1.25])

# --- Shared Legend ---
handles, plot_labels = ax1.get_legend_handles_labels()
fig.legend(handles, plot_labels, frameon=False, loc='upper center',
           bbox_to_anchor=(0.5, 1.1), ncol=4, fontsize=15)

plt.tight_layout()
fig.subplots_adjust(top=0.88)

save_name = 'Time_Plot_Combined.png'
plt.savefig(save_name, dpi=300, bbox_inches='tight')
plt.show()
print(f"Plot saved successfully as {save_name}")
