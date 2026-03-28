"""
monte_carlo_animation.py
─────────────────────────────────────────────────────────────────────────────
Monte Carlo uncertainty propagation for the UHPC closed-form flexural model.

Tension model parameters are sampled from ±2σ truncated normal distributions.
The animation streams curves in one batch at a time, then reveals the mean
and ±2σ band once all samples are drawn.

Output: monte_carlo_uhpc.mp4  (and an interactive window)
─────────────────────────────────────────────────────────────────────────────
"""

import warnings
warnings.filterwarnings('ignore')   # suppress divide-by-zero in unused stages

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.stats import truncnorm
from parametric_uhpc import run_full_model

np.random.seed(42)

# ── Base parameters (Saqif et al. 2022 – R-4-2.0 beam) ──────────────────────
# Geometry / loading
BASE = dict(
    L       = 1092.0,   # span, mm
    b       = 101.0,    # width, mm
    h       = 203.0,    # depth, mm
    pointBend = 4,
    S2      = 254.0,    # load-point spacing, mm
    Lp      = 254.0,    # pre-localization plastic length, mm
    cLp     = 125.0,    # post-localization plastic length, mm
    cover   = 38.0,     # mm
    # Tension model
    E           = 45526.0,   # MPa
    epsilon_cr  = 0.00015,   # mm/mm
    sigma_t1    = 11.5,      # peak hardening stress, MPa
    sigma_t2    =  5.5,      # first softening stress, MPa
    sigma_t3    =  1.5,      # residual stress, MPa
    epsilon_t1  = 0.003,     # strain at peak
    epsilon_t2  = 0.020,     # strain at 2nd knee
    epsilon_t3  = 0.080,     # ultimate tensile strain
    # Compression model  (Ec must differ from E: xi=Ec/E=1 causes /0 in stage111)
    Ec       = 45526.0 * 1.01,
    sigma_cy = 200.0,   # MPa
    sigma_cu =  30.0,   # MPa
    ecu      = 0.025,
    # Reinforcement
    Es          = 200000.0,
    fsy         =  460.0,   # MPa
    fsu         =  670.0,   # MPa
    epsilon_su  =  0.14,    # mm/mm
    botDiameter = (3/8)*25.4,  # 9.525 mm
    botCount    = 2,
    topDiameter = 10.0,
    topCount    = 0,        # singly reinforced
    plot        = False,
)

# ── Tension model uncertainty: key → (CV,  display label) ───────────────────
#    All six shape parameters of the quadrilinear tension model are varied.
#    E and epsilon_cr are held deterministic (material-specific constants).
VARIED = {
    'sigma_t1':   (0.10, r'$\sigma_{t1}$  (peak, MPa)'),
    'sigma_t2':   (0.15, r'$\sigma_{t2}$  (softening, MPa)'),
    'sigma_t3':   (0.20, r'$\sigma_{t3}$  (residual, MPa)'),
    'epsilon_t1': (0.10, r'$\varepsilon_{t1}$  (strain at peak)'),
    'epsilon_t2': (0.15, r'$\varepsilon_{t2}$  (2nd knee strain)'),
    'epsilon_t3': (0.20, r'$\varepsilon_{t3}$  (ultimate strain)'),
}

N_SAMPLES  = 1000
A_CLIP, B_CLIP = -2.0, 2.0   # ±2σ hard truncation

# ── Sampling ─────────────────────────────────────────────────────────────────
def sample_valid():
    """Rejection sample until physical ordering is satisfied."""
    while True:
        p = BASE.copy()
        for key, (cv, _) in VARIED.items():
            mu    = BASE[key]
            sigma = mu * cv
            p[key] = float(truncnorm.rvs(A_CLIP, B_CLIP, loc=mu, scale=sigma))
        ok = (
            p['sigma_t1'] > p['sigma_t2'] > p['sigma_t3'] > 0.0
            and p['epsilon_t1'] < p['epsilon_t2'] < p['epsilon_t3']
        )
        if ok:
            return p

# ── Run all simulations ───────────────────────────────────────────────────────
import time as _time
print(f"Running {N_SAMPLES} Parametric Closed-Form Model..")
all_params, results = [], []
_t0 = _time.perf_counter()
for i in range(N_SAMPLES):
    p = sample_valid()
    try:
        r = run_full_model(**p)
        all_params.append(p)
        results.append(r)
    except Exception:
        pass
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{N_SAMPLES}")
_sim_elapsed = _time.perf_counter() - _t0

N = len(results)
ms_per_run = (_sim_elapsed / N * 1000.0) if N else float('nan')
print(f"Completed {N} simulations in {_sim_elapsed:.2f}s  ({ms_per_run:.1f} ms/run)\n")

# ── Build tension stress-strain curves for animation ─────────────────────────
def tension_curve(p):
    ecr  = p['epsilon_cr']
    scr  = p['E'] * ecr
    eps  = [0.0, ecr, p['epsilon_t1'], p['epsilon_t2'], p['epsilon_t3']]
    sig  = [0.0, scr, p['sigma_t1'],   p['sigma_t2'],   p['sigma_t3']]
    return np.array(eps), np.array(sig)

def _as_step_series(value, target_len=None):
    try:
        arr = np.asarray(value, dtype=float)
    except Exception:
        return None

    if arr.size == 0:
        return None

    if arr.ndim == 1:
        out = arr
    elif arr.ndim == 2:
        if target_len is not None and arr.shape[0] == target_len:
            out = np.nanmax(arr, axis=1)
        elif target_len is not None and arr.shape[1] == target_len:
            out = np.nanmax(arr, axis=0)
        else:
            return None
    else:
        return None

    if target_len is not None and out.size != target_len:
        return None

    return np.asarray(out, dtype=float)

def compute_bottom_rebar_strain(result, p):
    beta = _as_step_series(result.get('beta'))
    curvature = _as_step_series(result.get('curvature'))
    if beta is None or curvature is None or beta.size != curvature.size:
        return None

    alpha = (p['h'] - p['cover']) / p['h']
    strain_bot = beta * p['epsilon_cr']
    with np.errstate(divide='ignore', invalid='ignore'):
        na_from_bot = strain_bot / curvature
        k_from_top = 1.0 - (na_from_bot / p['h'])
        est_bot = (-alpha + k_from_top) * beta * p['epsilon_cr'] / (k_from_top - 1.0)

    est_bot = np.asarray(est_bot, dtype=float)
    est_bot[~np.isfinite(est_bot)] = np.nan
    return est_bot

def load_at_rebar_yield(result, p):
    load = _as_step_series(result.get('load'), target_len=None)
    if load is None or load.size < 2:
        return np.nan, False

    strain = compute_bottom_rebar_strain(result, p)
    if strain is None or strain.size != load.size:
        return np.nan, False

    load = load / 1000.0
    yield_strain = p['fsy'] / p['Es']

    mask = np.isfinite(load) & np.isfinite(strain)
    load = load[mask]
    strain = strain[mask]
    if load.size < 2:
        return np.nan, False

    hits = np.flatnonzero(strain >= yield_strain)
    if hits.size == 0:
        return np.nan, False

    idx = int(hits[0])
    if idx == 0:
        return float(load[0]), True

    e0, e1 = strain[idx - 1], strain[idx]
    l0, l1 = load[idx - 1], load[idx]
    if not np.isfinite(e0) or not np.isfinite(e1) or np.isclose(e1, e0):
        return float(load[idx]), True

    frac = np.clip((yield_strain - e0) / (e1 - e0), 0.0, 1.0)
    return float(l0 + frac * (l1 - l0)), True

def build_hist_bins(values, n_bins=18):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.linspace(0.0, 1.0, n_bins + 1)

    lo = float(np.nanmin(values))
    hi = float(np.nanmax(values))
    if np.isclose(lo, hi):
        pad = max(abs(lo) * 0.05, 0.25)
    else:
        pad = (hi - lo) * 0.08

    return np.linspace(lo - pad, hi + pad, n_bins + 1)

def histogram_ceiling(values, bin_edges):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 1.0

    counts, _ = np.histogram(values, bins=bin_edges)
    ymax = float(max(np.nanmax(counts), 1.0))

    if values.size > 1:
        sigma = float(np.nanstd(values, ddof=1))
        if sigma > 0.0:
            mu = float(np.nanmean(values))
            x = np.linspace(bin_edges[0], bin_edges[-1], 256)
            bin_width = float(np.mean(np.diff(bin_edges)))
            curve = (
                values.size * bin_width
                * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
                / (sigma * np.sqrt(2.0 * np.pi))
            )
            ymax = max(ymax, float(np.nanmax(curve)))

    return ymax * 1.25

def reset_histogram(bars, mean_line, fit_line, stat_text):
    for bar in bars:
        bar.set_height(0.0)
    mean_line.set_alpha(0.0)
    fit_line.set_data([], [])
    stat_text.set_text('')

def update_histogram(values, bars, bin_edges, mean_line, fit_line, stat_text,
                     units='kN', extra_text=''):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    counts, _ = np.histogram(values, bins=bin_edges)
    for bar, count in zip(bars, counts):
        bar.set_height(float(count))

    lines = []
    if values.size == 0:
        mean_line.set_alpha(0.0)
        fit_line.set_data([], [])
        lines.append('n = 0')
    else:
        mu = float(np.nanmean(values))
        sigma = float(np.nanstd(values, ddof=1)) if values.size > 1 else 0.0

        mean_line.set_xdata([mu, mu])
        mean_line.set_alpha(0.95)

        lines.append(f'n = {values.size}')
        lines.append(f'mu = {mu:.2f} {units}')
        lines.append(f'sigma = {sigma:.2f} {units}')

        if values.size > 1 and sigma > 0.0:
            x = np.linspace(bin_edges[0], bin_edges[-1], 256)
            bin_width = float(np.mean(np.diff(bin_edges)))
            curve = (
                values.size * bin_width
                * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
                / (sigma * np.sqrt(2.0 * np.pi))
            )
            fit_line.set_data(x, curve)
        else:
            fit_line.set_data([], [])

    if extra_text:
        lines.append(extra_text)

    stat_text.set_text('\n'.join(lines))
    return list(bars) + [mean_line, fit_line, stat_text]

t_curves = [tension_curve(p) for p in all_params]

# ── Build load-deflection arrays on a common axis ────────────────────────────
trimmed      = []
valid_params = []
valid_curves = []
peak_loads   = []
yield_loads  = []

for r, p, tc in zip(results, all_params, t_curves):
    d = np.array(r['deflection'])
    l = np.array(r['load']) / 1000.0   # N → kN
    if not np.any(np.isfinite(l)) or np.nanmax(l) <= 0:
        continue
    # keep full curve including post-peak softening
    mask = np.isfinite(d) & np.isfinite(l) & (d > 0)
    d_tr, l_tr = d[mask], l[mask]
    if len(d_tr) == 0 or d_tr[-1] <= 0:
        continue
    trimmed.append((d_tr, l_tr))
    valid_params.append(p)
    valid_curves.append(tc)
    peak_loads.append(float(np.nanmax(l_tr)))

    yield_load, yielded = load_at_rebar_yield(r, p)
    yield_loads.append(yield_load if yielded else np.nan)

all_params = valid_params
t_curves   = valid_curves
N          = len(trimmed)
peak_loads = np.asarray(peak_loads, dtype=float)
yield_loads = np.asarray(yield_loads, dtype=float)
print(f"Valid runs after filtering: {N}")

if N == 0:
    raise RuntimeError("No valid Monte Carlo runs remained after filtering.")

d_peaks  = [d[-1] for d, _ in trimmed]
d_end    = np.nanmax(d_peaks)
d_common = np.linspace(0.0, d_end, 700)

loads_grid = np.array([
    np.interp(d_common, d_tr, l_tr, left=0.0, right=l_tr[-1])
    for d_tr, l_tr in trimmed
])

mean_load = np.nanmean(loads_grid, axis=0)
std_load  = np.nanstd(loads_grid,  axis=0)
yield_strain = BASE['fsy'] / BASE['Es']
yield_hist_seed = yield_loads if np.any(np.isfinite(yield_loads)) else peak_loads
yield_hist_bins = build_hist_bins(yield_hist_seed)
peak_hist_bins = build_hist_bins(peak_loads)
yield_hist_ymax = histogram_ceiling(yield_loads, yield_hist_bins)
peak_hist_ymax = histogram_ceiling(peak_loads, peak_hist_bins)
print(" ")

# ── Color map by sigma_t1 (peak hardening stress) ────────────────────────────
s1_arr  = np.array([p['sigma_t1'] for p in all_params])
norm_cm = Normalize(vmin=s1_arr.min(), vmax=s1_arr.max())
cmap    = plt.cm.plasma

# ── Figure / axes ─────────────────────────────────────────────────────────────
BG      = '#0d0d0d'
PANEL   = '#141414'
GRIDC   = '#252525'
TXTC    = '#cccccc'
LBLC    = '#eeeeee'

fig = plt.figure(figsize=(16, 10), facecolor=BG)
gs = fig.add_gridspec(2, 2, height_ratios=[2.2, 1.15])
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])
fig.subplots_adjust(left=0.07, right=0.95, top=0.90, bottom=0.08, wspace=0.26, hspace=0.30)

for ax in (ax1, ax2, ax3, ax4):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TXTC, labelsize=10)
    ax.xaxis.label.set_color(LBLC)
    ax.yaxis.label.set_color(LBLC)
    ax.title.set_color('white')
    for sp in ax.spines.values():
        sp.set_edgecolor('#333333')
    ax.grid(True, color=GRIDC, linestyle='--', linewidth=0.5, alpha=0.7)

# Axis ranges
t_xmax = BASE['epsilon_t3'] * 1.2
t_ymax = BASE['sigma_t1']   * 1.9
d_xmax = d_end * 1.08
l_ymax = np.nanmax(loads_grid) * 1.2

ax1.set_xlim(0, t_xmax);  ax1.set_ylim(0, t_ymax)
ax2.set_xlim(0, d_xmax);  ax2.set_ylim(0, l_ymax)

ax1.set_xlabel("Tensile Strain  (mm/mm)", fontsize=12)
ax1.set_ylabel("Tensile Stress  (MPa)",   fontsize=12)
ax1.set_title("Input UHPC Tension Model",
              fontsize=13, pad=8, color='white')

ax2.set_xlabel("Midspan Deflection  (mm)", fontsize=12)
ax2.set_ylabel("Applied Load  (kN)",       fontsize=12)
ax2.set_title("Output Beam Load-Deflection (R-UHPC)",
              fontsize=13, pad=8, color='white')

ax3.set_xlim(yield_hist_bins[0], yield_hist_bins[-1])
ax3.set_ylim(0, yield_hist_ymax)
ax3.set_xlabel("Load at Rebar Yield  (kN)", fontsize=11)
ax3.set_ylabel("Count", fontsize=11)
ax3.set_title("Load at Rebar Yield  (kN)\nHistogram",
              fontsize=12, pad=8, color='white')

ax4.set_xlim(peak_hist_bins[0], peak_hist_bins[-1])
ax4.set_ylim(0, peak_hist_ymax)
ax4.set_xlabel("Peak Load  (kN)", fontsize=11)
ax4.set_ylabel("Count", fontsize=11)
ax4.set_title("Peak Load  (kN)\nHistogram",
              fontsize=12, pad=8, color='white')

# Pre-create mean + band artists (initially invisible)
mean_ld,  = ax2.plot([], [], color='white',  lw=2.5,  zorder=12, label='Mean',   alpha=0)
upper_ld, = ax2.plot(d_common, mean_load + 2*std_load, color='#00e5ff', lw=1.2,
                     ls='--', zorder=11, label='±2σ', alpha=0)
lower_ld, = ax2.plot(d_common, mean_load - 2*std_load, color='#00e5ff', lw=1.2,
                     ls='--', zorder=11, alpha=0)
fill_art  = ax2.fill_between(d_common,
                             mean_load + 2*std_load,
                             mean_load - 2*std_load,
                             color='#00e5ff', alpha=0, zorder=10)

mean_t,   = ax1.plot([], [], color='white', lw=2.5, zorder=12, alpha=0)

yield_centers = 0.5 * (yield_hist_bins[:-1] + yield_hist_bins[1:])
yield_widths = np.diff(yield_hist_bins)
yield_bars = ax3.bar(
    yield_centers, np.zeros_like(yield_centers), width=yield_widths * 0.9,
    color='#00e5ff', alpha=0.22, edgecolor='#7be7ff', linewidth=1.0, zorder=3
)
yield_mean_line = ax3.axvline(yield_centers.mean(), color='white', lw=1.6, alpha=0.0, zorder=5)
yield_fit, = ax3.plot([], [], color='#7be7ff', lw=1.4, zorder=4)
yield_stats = ax3.text(
    0.03, 0.96, '', transform=ax3.transAxes, ha='left', va='top',
    fontsize=9, color='white',
    bbox=dict(boxstyle='round,pad=0.3', fc='#1f1f1f', ec='#444')
)

peak_centers = 0.5 * (peak_hist_bins[:-1] + peak_hist_bins[1:])
peak_widths = np.diff(peak_hist_bins)
peak_bars = ax4.bar(
    peak_centers, np.zeros_like(peak_centers), width=peak_widths * 0.9,
    color='#ffb347', alpha=0.24, edgecolor='#ffd08a', linewidth=1.0, zorder=3
)
peak_mean_line = ax4.axvline(peak_centers.mean(), color='white', lw=1.6, alpha=0.0, zorder=5)
peak_fit, = ax4.plot([], [], color='#ffd08a', lw=1.4, zorder=4)
peak_stats = ax4.text(
    0.03, 0.96, '', transform=ax4.transAxes, ha='left', va='top',
    fontsize=9, color='white',
    bbox=dict(boxstyle='round,pad=0.3', fc='#1f1f1f', ec='#444')
)

# Counter / title annotation
counter   = ax1.text(0.97, 0.96, '', transform=ax1.transAxes,
                     ha='right', va='top', fontsize=14,
                     color='white', fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.3', fc='#1f1f1f', ec='#444'))

fig.text(0.5, 0.96,
         f"Parametric R-UHPC Model |  "
         f"{_sim_elapsed:.1f}s for {N} runs  ({ms_per_run:.1f} ms/run)",
         ha='center', va='top', fontsize=11, color='#888888')

# Colorbar
sm   = ScalarMappable(norm=norm_cm, cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax1, pad=0.03, fraction=0.044)
cbar.set_label(r'Peak Tensile Stress $\sigma_{t1}$  (MPa)',
               color=TXTC, fontsize=10)
cbar.ax.tick_params(colors=TXTC)
plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TXTC)

# ── Animation bookkeeping ─────────────────────────────────────────────────────
t_lines, ld_lines = [], []
BATCH           = 4
N_DATA_FRAMES   = (N + BATCH - 1) // BATCH
N_REVEAL_FRAMES = 20
N_HOLD_FRAMES   = 30
N_TOTAL         = N_DATA_FRAMES + N_REVEAL_FRAMES + N_HOLD_FRAMES

# Precomputed elapsed-time stamp for each sample (stopwatch values)
sim_times = np.linspace(0.0, _sim_elapsed, N + 1)   # sim_times[i] = time when sample i was added

legend_added = [False]

def init():
    mean_ld.set_data([], [])
    mean_ld.set_alpha(0)
    upper_ld.set_alpha(0)
    lower_ld.set_alpha(0)
    fill_art.set_alpha(0)
    mean_t.set_data([], [])
    mean_t.set_alpha(0)
    counter.set_text('')
    reset_histogram(yield_bars, yield_mean_line, yield_fit, yield_stats)
    reset_histogram(peak_bars, peak_mean_line, peak_fit, peak_stats)
    return []

def update(frame):
    drawn = min(frame * BATCH, N)

    # ── Stream new tension model curves ──────────────────────────────────────
    while len(t_lines) < drawn:
        i   = len(t_lines)
        col = cmap(norm_cm(s1_arr[i]))
        ln, = ax1.plot(t_curves[i][0], t_curves[i][1],
                       color=col, alpha=0.18, lw=0.85, zorder=2)
        t_lines.append(ln)

    # ── Stream new load-deflection curves ────────────────────────────────────
    while len(ld_lines) < drawn:
        i   = len(ld_lines)
        col = cmap(norm_cm(s1_arr[i]))
        ln, = ax2.plot(d_common, loads_grid[i],
                       color=col, alpha=0.15, lw=0.85, zorder=2)
        ld_lines.append(ln)

    elapsed_shown = sim_times[drawn] if drawn > 0 else 0.0
    counter.set_text(f'n = {drawn} / {N}    {elapsed_shown:.2f}s')

    yielded_now = int(np.count_nonzero(np.isfinite(yield_loads[:drawn])))
    yield_extra = f'yielded = {yielded_now} / {drawn}\neps_y = {yield_strain:.5f}'
    yield_hist_artists = update_histogram(
        yield_loads[:drawn], yield_bars, yield_hist_bins,
        yield_mean_line, yield_fit, yield_stats,
        units='kN', extra_text=yield_extra
    )
    peak_hist_artists = update_histogram(
        peak_loads[:drawn], peak_bars, peak_hist_bins,
        peak_mean_line, peak_fit, peak_stats,
        units='kN', extra_text=f'drawn = {drawn} / {N}'
    )

    # ── Reveal mean + band after all data is drawn ───────────────────────────
    if frame >= N_DATA_FRAMES:
        t_r = min((frame - N_DATA_FRAMES) / max(N_REVEAL_FRAMES, 1), 1.0)

        # Mean load-deflection
        mean_ld.set_data(d_common, mean_load)
        mean_ld.set_alpha(t_r)

        # ±2σ lines
        upper_ld.set_alpha(t_r * 0.85)
        lower_ld.set_alpha(t_r * 0.85)

        # ±2σ fill band (PolyCollection supports set_alpha)
        fill_art.set_alpha(t_r * 0.10)

        # Mean tension model
        ecr = BASE['epsilon_cr']
        mean_t.set_data(
            [0.0, ecr, BASE['epsilon_t1'], BASE['epsilon_t2'], BASE['epsilon_t3']],
            [0.0, BASE['E']*ecr, BASE['sigma_t1'], BASE['sigma_t2'], BASE['sigma_t3']],
        )
        mean_t.set_alpha(t_r)

        # Add legend once
        if not legend_added[0] and t_r >= 1.0:
            ax2.legend(handles=[mean_ld, upper_ld],
                       fontsize=10, facecolor='#1a1a1a',
                       edgecolor='#444', labelcolor='white',
                       loc='upper right')
            legend_added[0] = True

    return (
        t_lines + ld_lines
        + [mean_ld, upper_ld, lower_ld, mean_t, counter]
        + yield_hist_artists + peak_hist_artists
    )

ani = animation.FuncAnimation(
    fig, update,
    frames   = N_TOTAL,
    init_func= init,
    interval = 45,       # ms between frames → ~22 fps preview
    blit     = False,
    repeat   = False,
)

# ── Save ─────────────────────────────────────────────────────────────────────
OUT = r'C:\Users\devan\Projects\HRC_Opensource\monte_carlo_uhpc.mp4'
print(f"Rendering {N_TOTAL} frames -> {OUT} ...")
writer = animation.FFMpegWriter(
    fps        = 24,
    bitrate    = 5000,
    extra_args = ['-pix_fmt', 'yuv420p'],   # broad compatibility
)
ani.save(OUT, writer=writer, dpi=150,
         savefig_kwargs={'facecolor': BG})
print(f"Saved:  {OUT}")

plt.show()
