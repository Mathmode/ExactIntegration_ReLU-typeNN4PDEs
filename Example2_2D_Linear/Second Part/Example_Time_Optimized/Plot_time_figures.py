"""Figures of the full-epoch timing. Run after the three Main_time_fast.py scripts (CMC, Exact, Monte Carlo).

  Figure 1, full_epoch_fixed_sizes.png
      Time per epoch of FB, NW and DNW with r = rbar = 10, for CMC and exact integration, and of standard
      Monte Carlo (n_MC = 10^4), against the number of neurons n (16 to 1024).
  Figure 2, speedup_alpha_beta.png
      Speed-up over FB, T_FB / T per epoch, of NW (r = alpha n) and DNW (r = alpha n, rbar = beta n) for
      (alpha, beta) = (0.5, 0.5) and (0.5, 0.25), against n (16 to 1024), for CMC and exact integration.
      Dotted lines: the pair-count prediction 1/alpha for NW and 1/(2 alpha beta) for DNW.

Reads the JSON files written by the timing scripts and writes the figures next to this file.
Run:  python Plot_time_figures.py
"""
import os, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator

My_Blue = (0, 0/255, 102/255)
My_Orange = (255/255, 178/255, 102/255)
My_Green = (10/255, 102/255, 28/255)

HERE = os.path.dirname(os.path.abspath(__file__))   # Example_Time_Optimized (this script's folder)
SOURCES = {
    'cmc': os.path.join(HERE, 'Deep_Ritz_2D_CMC_time_report', 'timing_results_CMC_full_epoch.json'),
    'exact': os.path.join(HERE, 'Deep_Ritz_2D_Exact_time_report', 'timing_results_Exact_full_epoch.json'),
    'mc': os.path.join(HERE, 'Deep_Ritz_2D_MonteCarlo_time_report', 'timing_results_MC_full_epoch.json'),
}
COLOR = {'cmc': My_Blue, 'exact': My_Orange, 'mc': 'red'}
METHOD_COLOR = {'FB': My_Blue, 'NW_r10': My_Orange, 'DNW_r10': My_Green}   # colour = method, as in the legend
MARKER = {'FB': 'o', 'NW_r10': 's', 'DNW_r10': '^'}
# C_NW = 'tab:orange'
# C_DNW = {(0, 0/255, 102/255): 'tab:blue', (0.5, 0.25): 'tab:purple'}



def load():
    data = {}
    for key, path in SOURCES.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"missing {path}: run the timing script first")
        with open(path) as f:
            data[key] = json.load(f)
        if 'by_n' in data[key]:
            missing = [N for N in data[key]['neurons'] if str(N) not in data[key]['by_n']]
            if missing:
                raise ValueError(f"{path} has no results for n = {missing}: the timing run did not finish. "
                                 "Rerun that Main_time_fast.py with nothing else running.")
    return data


def figure_fixed_sizes(data):
    font = {'family': 'Times',
            'weight': 'normal',
            'size': 18}
    plt.rc('font', **font)
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = r'\usepackage{mathrsfs} \usepackage{amssymb}'

    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    # colour = method; exact integration: solid with filled markers; CMC: dashed with open markers
    for key in ('exact', 'cmc'):
        block = data[key]
        n = np.array(block['neurons'], float)
        for name in ('FB', 'NW_r10', 'DNW_r10'):
            t = np.array([block['by_n'][str(N)][name] for N in block['neurons']])
            exact = key == 'exact'
            ax.loglog(n, t, color=METHOD_COLOR[name], marker=MARKER[name], markersize=9,
                      linestyle='-' if exact else '--', linewidth=2,
                      markerfacecolor=METHOD_COLOR[name] if exact else 'none')
    mc = data['mc']
    ax.loglog(np.array(mc['neurons'], float), np.array(mc['SMC']), color=COLOR['mc'], marker='D',
              markersize=9, linestyle=':', linewidth=2)
    ax.set_xlabel('Number of neurons $n$', fontsize=18)
    ax.set_ylabel('Time per epoch (s)', fontsize=18)
    ax.set_title(r'Full epoch time with fixed sizes $r=\bar r=10$',fontsize=18)
    ax.grid(True, which='both', alpha=0.3)
    # reference lines y = c n^2 and y = c n on the log-log axes, scaled to pass through the CMC curves at n = 1024
    anchor = data['cmc']['by_n'][str(data['cmc']['neurons'][-1])]
    n_ref = np.array([16.0, 2048])
    c2 = anchor['FB'] / 1024 ** 2            # y = c2 n^2 passes through the CMC FB point
    c1 = anchor['NW_r10'] / 1024             # y = c1 n passes through the CMC NW (r = 10) point
    ax.loglog(n_ref, c2 * n_ref ** 2, color='k', linestyle='--', lw=1.2)
    ax.loglog(n_ref, c1 * n_ref, color='k', linestyle=':', lw=1.2)

    # legend in the layout of the reference picture: one block per group, bold group titles
    def sample(method, filled, linestyle, label):
        colour = METHOD_COLOR[method]
        return Line2D([], [], color=colour, marker=MARKER[method], linestyle=linestyle, markersize=6,
                      markerfacecolor=colour if filled else 'none', label=label)

    names = [('FB', 'Full-Batched'), ('NW_r10', 'Neuron-Wise'), ('DNW_r10', 'Double Neuron-Wise')]
    exact_handles = [sample(m, True, '-', lab) for m, lab in names]
    cmc_handles = [sample(m, False, '--', lab) for m, lab in names]
    mc_handles = [Line2D([], [], color=COLOR['mc'], marker='D', linestyle=':', markersize=6,
                         markerfacecolor=COLOR['mc'], label='Standard Monte Carlo')]
    ref_handles = [Line2D([], [], color='k', linestyle='--', lw=1.2, label=r'$y \propto n^2$'),
                   Line2D([], [], color='k', linestyle=':', lw=1.2, label=r'$y \propto n$')]
    kw = dict(frameon=False, loc='upper left', borderaxespad=-1.0, handlelength=1.6, handletextpad=.6,
              fontsize=18, title_fontproperties={'weight': 'bold', 'size': 18})
    fig.legend(handles=exact_handles, title=r'\textbf{Exact Integration}', bbox_to_anchor=(0.02, 0.995), **kw)
    fig.legend(handles=cmc_handles, title=r'\textbf{Conforming Monte Carlo}', bbox_to_anchor=(0.37, 0.995), **kw)
    # the reference slopes sit under the Standard MC entry, in the same block
    fig.legend(handles=mc_handles + ref_handles, title=r'\textbf{}', bbox_to_anchor=(0.72, 0.995), **kw, )
    for leg in fig.legends:                      # bold group titles
        if leg.get_title().get_text():
            leg.get_title().set_fontweight('bold')
    fig.subplots_adjust(left=0.09, right=0.98, top=0.74, bottom=0.12)
    path = os.path.join(HERE, 'full_epoch_fixed_sizes.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    return fig, path


# def figure_speedup(data):
#     fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharey=True)
#     for ax, key, title in ((axes[0], 'cmc', 'CMC'), (axes[1], 'exact', 'Exact integration')):
#         block = data[key]
#         neurons = block['neurons']
#         n = np.array(neurons, float)

#         def t(N, name):
#             return block['by_n'][str(N)][name]

#         for a in sorted({a for a, _ in block['settings']}):
#             s = [t(N, 'FB') / t(N, f'NW_a{a}') for N in neurons]
#             ax.plot(n, s, 's-', color=C_NW, ms=4, label=rf'NW, $r={a:g}n$')
#             ax.axhline(1 / a, color=C_NW, ls=':', lw=1.2)
#         for a, b in block['settings']:
#             s = [t(N, 'FB') / t(N, f'DNW_a{a}_b{b}') for N in neurons]
#             ax.plot(n, s, 'o-', color=C_DNW[(a, b)], ms=4, label=rf'DNW, $r={a:g}n$, $\bar r={b:g}n$')
#             ax.axhline(1 / (2 * a * b), color=C_DNW[(a, b)], ls=':', lw=1.2)
#         ax.axhline(1.0, color='gray', lw=0.6)
#         ax.set_xscale('log', base=2)
#         ax.set_yscale('log')
#         ax.set_xticks([16, 64, 256, 1024])
#         ax.set_xticklabels(['16', '64', '256', '1024'])
#         ax.xaxis.set_minor_locator(NullLocator())
#         ax.set_xlabel('number of neurons $n$')
#         ax.set_title(title)
#         ax.grid(True, which='major', alpha=0.3)
#         ax.legend(fontsize=8)
#     axes[0].set_ylabel(r'speed-up over FB, $T_{\rm FB}/T$')
#     fig.suptitle('Full epoch speed-up over FB (dotted: pair-count prediction $1/\\alpha$ for NW, '
#                  '$1/(2\\alpha\\beta)$ for DNW)', fontsize=10)
#     fig.tight_layout(rect=(0, 0, 1, 0.95))
#     path = os.path.join(HERE, 'speedup_alpha_beta.png')
#     fig.savefig(path, dpi=150)
#     return fig, path


def print_speedups(block, title):
    settings = [tuple(s) for s in block['settings']]
    names = [f'NW_a{a}' for a in sorted({a for a, _ in settings})] + [f'DNW_a{a}_b{b}' for a, b in settings]
    print(f"{title}: time per epoch and speed-up T_FB / T")
    print(f"{'n':>6}{'FB [s]':>11}" + "".join(f"{nm:>18}" for nm in names))
    for N in block['neurons']:
        t = block['by_n'][str(N)]
        print(f"{N:>6}{t['FB']:>11.4f}" + "".join(f"{t['FB'] / t[nm]:>18.2f}" for nm in names))


if __name__ == "__main__":
    data = load()
    _, p1 = figure_fixed_sizes(data)
    # _, p2 = figure_speedup(data)
    print_speedups(data['cmc'], 'CMC')
    print_speedups(data['exact'], 'Exact integration')
    print("figures saved:\n   " + "\n   ".join([p1]))
    plt.show()
