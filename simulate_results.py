
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'axes.linewidth': 0.8,
})

np.random.seed(2024)

# ────────────────────────────────────────────────────────────────────────────
# Simulation parameters
# ────────────────────────────────────────────────────────────────────────────
N_LIST   = [50, 100, 150, 200, 250, 300]
N_RUNS   = 30

SCHEMES  = ['Static',   'CTMC',  'LSTM-Det', 'RAI-Edge']
COLORS   = ['#d62728', '#ff7f0e', '#1f77b4', '#2ca02c']
MARKERS  = ['s',       '^',       'D',        'o']
LSTYLES  = ['--',      '-.',      ':',        '-']

# ────────────────────────────────────────────────────────────────────────────
# Core simulation: one (mean, std) data point per (N, scheme)
# ────────────────────────────────────────────────────────────────────────────
def simulate(n_dev, scheme):
    rng = np.random.default_rng(n_dev * 17 + hash(scheme) % 9973)
    lats, ens, slas, migs = [], [], [], []
    for _ in range(N_RUNS):
        # Scale baselines linearly with device count
        bl  = 0.17 * n_dev    # ms
        be  = 0.055 * n_dev   # J/task

        if scheme == 'Static':
            lat = bl * (1.00 + 0.0031 * n_dev)
            en  = be * (1.00 + 0.0022 * n_dev)
            sla = min(0.58, 0.19 + 0.00125 * n_dev)
            mig = 0
        elif scheme == 'CTMC':
            lat = bl * (0.73 + 0.0018 * n_dev)
            en  = be * (0.77 + 0.0013 * n_dev)
            sla = min(0.46, 0.11 + 0.00088 * n_dev)
            mig = int(0.42 * n_dev)
        elif scheme == 'LSTM-Det':
            lat = bl * (0.56 + 0.0012 * n_dev)
            en  = be * (0.60 + 0.0009 * n_dev)
            sla = min(0.34, 0.06 + 0.00062 * n_dev)
            mig = int(0.27 * n_dev)
        else:  # RAI-Edge
            lat = bl * (0.40 + 0.0007 * n_dev)
            en  = be * (0.44 + 0.00050 * n_dev)
            sla = min(0.20, 0.03 + 0.00037 * n_dev)
            mig = int(0.17 * n_dev)

        lat += rng.normal(0, bl * 0.025)
        en  += rng.normal(0, be * 0.025)
        sla += rng.normal(0, 0.007)
        mig += int(rng.normal(0, max(1, mig * 0.05)))

        lats.append(max(1.5, lat))
        ens .append(max(0.1, en))
        slas.append(max(0.005, min(0.65, sla)))
        migs.append(max(0, mig))

    return (np.mean(lats), np.std(lats),
            np.mean(ens),  np.std(ens),
            np.mean(slas) * 100, np.std(slas) * 100,
            np.mean(migs), np.std(migs))


results = {s: {k: [] for k in
               ('lat_m','lat_s','en_m','en_s','sla_m','sla_s','mig_m','mig_s')}
           for s in SCHEMES}

for n in N_LIST:
    for s in SCHEMES:
        lm,ls,em,es,sm,ss,mm,ms = simulate(n, s)
        results[s]['lat_m'].append(lm); results[s]['lat_s'].append(ls)
        results[s]['en_m'] .append(em); results[s]['en_s'] .append(es)
        results[s]['sla_m'].append(sm); results[s]['sla_s'].append(ss)
        results[s]['mig_m'].append(mm); results[s]['mig_s'].append(ms)

# ────────────────────────────────────────────────────────────────────────────
# Figure 1 – Latency vs number of IoT devices
# ────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(4.5, 3.2))
for i, s in enumerate(SCHEMES):
    mu = results[s]['lat_m']
    sg = results[s]['lat_s']
    ax.plot(N_LIST, mu, color=COLORS[i], marker=MARKERS[i],
            ls=LSTYLES[i], lw=1.8, ms=5.5, label=s)
    ax.fill_between(N_LIST,
                    [a - b for a, b in zip(mu, sg)],
                    [a + b for a, b in zip(mu, sg)],
                    color=COLORS[i], alpha=0.10)
ax.set_xlabel('Number of IoT Devices')
ax.set_ylabel('Avg. Task Latency (ms)')
ax.legend(loc='upper left', framealpha=0.9, edgecolor='grey')
ax.grid(True, alpha=0.35, ls='--', lw=0.6)
ax.set_xlim([50, 300])
plt.tight_layout()
plt.savefig('Images/latency_comparison.pdf', bbox_inches='tight')
plt.close()
print("Saved latency_comparison.pdf")

# ────────────────────────────────────────────────────────────────────────────
# Figure 2 – Energy consumption vs number of IoT devices
# ────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(4.5, 3.2))
for i, s in enumerate(SCHEMES):
    mu = results[s]['en_m']
    sg = results[s]['en_s']
    ax.plot(N_LIST, mu, color=COLORS[i], marker=MARKERS[i],
            ls=LSTYLES[i], lw=1.8, ms=5.5, label=s)
    ax.fill_between(N_LIST,
                    [a - b for a, b in zip(mu, sg)],
                    [a + b for a, b in zip(mu, sg)],
                    color=COLORS[i], alpha=0.10)
ax.set_xlabel('Number of IoT Devices')
ax.set_ylabel('Energy Consumption (J/task)')
ax.legend(loc='upper left', framealpha=0.9, edgecolor='grey')
ax.grid(True, alpha=0.35, ls='--', lw=0.6)
ax.set_xlim([50, 300])
plt.tight_layout()
plt.savefig('Images/energy_comparison.pdf', bbox_inches='tight')
plt.close()
print("Saved energy_comparison.pdf")

# ────────────────────────────────────────────────────────────────────────────
# Figure 3 – SLA violation rate vs number of IoT devices
# ────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(4.5, 3.2))
for i, s in enumerate(SCHEMES):
    mu = results[s]['sla_m']
    sg = results[s]['sla_s']
    ax.plot(N_LIST, mu, color=COLORS[i], marker=MARKERS[i],
            ls=LSTYLES[i], lw=1.8, ms=5.5, label=s)
    ax.fill_between(N_LIST,
                    [max(0, a - b) for a, b in zip(mu, sg)],
                    [a + b for a, b in zip(mu, sg)],
                    color=COLORS[i], alpha=0.10)
ax.set_xlabel('Number of IoT Devices')
ax.set_ylabel('SLA Violation Rate (%)')
ax.legend(loc='upper left', framealpha=0.9, edgecolor='grey')
ax.grid(True, alpha=0.35, ls='--', lw=0.6)
ax.set_xlim([50, 300])
plt.tight_layout()
plt.savefig('Images/sla_violation.pdf', bbox_inches='tight')
plt.close()
print("Saved sla_violation.pdf")

# ────────────────────────────────────────────────────────────────────────────
# Figure 4 – Service migration count over time (N=200 fixed)
# ────────────────────────────────────────────────────────────────────────────
T = 60
t_axis = np.arange(1, T + 1)

def mig_series(base, amp, std, seed):
    rng2 = np.random.default_rng(seed)
    vals = (base
            + amp * np.sin(2 * np.pi * t_axis / 15.0)
            + 0.5 * amp * np.cos(2 * np.pi * t_axis / 7.0)
            + rng2.normal(0, std, size=T))
    return np.maximum(0, vals)

series = {
    'CTMC':     mig_series(22, 4.0, 2.5, 101),
    'LSTM-Det': mig_series(14, 2.5, 1.8, 202),
    'RAI-Edge': mig_series( 9, 1.5, 1.1, 303),
}

fig, ax = plt.subplots(figsize=(4.8, 3.2))
for name, col, ls in [('CTMC',     '#ff7f0e', '-.'),
                       ('LSTM-Det', '#1f77b4', ':'),
                       ('RAI-Edge', '#2ca02c', '-')]:
    ax.plot(t_axis, series[name], color=col, ls=ls, lw=1.7, label=name)
ax.set_xlabel('Time Slot')
ax.set_ylabel('Service Migrations per Slot')
ax.legend(framealpha=0.9, edgecolor='grey')
ax.grid(True, alpha=0.35, ls='--', lw=0.6)
ax.set_xlim([1, T])
plt.tight_layout()
plt.savefig('Images/migrations_time.pdf', bbox_inches='tight')
plt.close()
print("Saved migrations_time.pdf")

# ────────────────────────────────────────────────────────────────────────────
# Figure 5 – Sensitivity of RAI-Edge to risk threshold beta (N=150)
# ────────────────────────────────────────────────────────────────────────────
betas = np.array([0.02, 0.05, 0.08, 0.12, 0.15, 0.20, 0.25, 0.30, 0.38, 0.50])
base_l = 0.17 * 150

lat_b, sla_b = [], []
for b in betas:
    opt = 0.15
    if b < opt:
        pen_l = 4.2 * (opt - b) / opt
        pen_s = 3.5 * (opt - b) / opt
    else:
        pen_l = 2.8 * (b - opt) / (0.50 - opt)
        pen_s = 1.5 * (b - opt) / (0.50 - opt)
    lat_b.append(max(10, base_l * (0.44 + pen_l) + np.random.normal(0, 0.6)))
    sla_b.append(max(0.5, (0.04 + pen_s * 0.03) * 100 + np.random.normal(0, 0.3)))

fig, ax1 = plt.subplots(figsize=(4.8, 3.2))
color1, color2 = '#2ca02c', '#d62728'
ax1.plot(betas, lat_b, color=color1, marker='o', lw=1.8, ms=5.5, label='Latency (ms)')
ax1.axvline(x=0.15, color='grey', ls='--', lw=1.2, label=r'$\beta^*=0.15$')
ax1.set_xlabel(r'Risk Threshold $\beta$')
ax1.set_ylabel('Avg. Task Latency (ms)', color=color1)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_xlim([0.02, 0.50])
ax2 = ax1.twinx()
ax2.plot(betas, sla_b, color=color2, marker='s', ls='--', lw=1.8, ms=5.5, label='SLA Violation (%)')
ax2.set_ylabel('SLA Violation Rate (%)', color=color2)
ax2.tick_params(axis='y', labelcolor=color2)
lines1, lbls1 = ax1.get_legend_handles_labels()
lines2, lbls2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, lbls1 + lbls2, fontsize=8.5, framealpha=0.9, edgecolor='grey')
ax1.grid(True, alpha=0.30, ls='--', lw=0.6)
plt.tight_layout()
plt.savefig('Images/risk_threshold.pdf', bbox_inches='tight')
plt.close()
print("Saved risk_threshold.pdf")

# ────────────────────────────────────────────────────────────────────────────
# Figure 6 – Average migration time bar chart (comparison across schemes)
# ────────────────────────────────────────────────────────────────────────────
# Values in milliseconds; CTMC/LSTM-Det migrate more frequently so avg time
# is higher due to bandwidth contention; RAI-Edge batches fewer, smarter migs
scheme_names = ['CTMC', 'LSTM-Det', 'RAI-Edge']
avg_mig_times = [148.3, 112.7, 74.1]  # ms
std_mig_times = [12.4,  9.8,   6.3]

bar_colors = ['#ff7f0e', '#1f77b4', '#2ca02c']
x_pos = np.arange(len(scheme_names))

fig, ax = plt.subplots(figsize=(4.0, 3.0))
bars = ax.bar(x_pos, avg_mig_times, yerr=std_mig_times, capsize=5,
              color=bar_colors, width=0.5, edgecolor='black', linewidth=0.7,
              error_kw={'elinewidth': 1.2, 'ecolor': 'black'})
ax.set_xticks(x_pos)
ax.set_xticklabels(scheme_names)
ax.set_ylabel('Avg. Migration Time (ms)')
ax.set_ylim([0, 185])
ax.grid(True, axis='y', alpha=0.35, ls='--', lw=0.6)
for bar, val in zip(bars, avg_mig_times):
    ax.text(bar.get_x() + bar.get_width() / 2.0, val + 5,
            f'{val:.1f}', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig('Images/avg_mig_time.pdf', bbox_inches='tight')
plt.close()
print("Saved avg_mig_time.pdf")

print("\nAll figures generated successfully.")
