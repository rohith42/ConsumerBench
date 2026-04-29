"""
Gantt chart with SLO attainment from a ConsumerBench results directory.

Usage:
    python gantt_chart.py <results_dir> [output.png]

Discovers all task_*_perf.log files in <results_dir>, skips dummy tasks,
computes SLO attainment per app type, and saves a Gantt chart PNG.
"""

import ast
import glob
import os
import re
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

plt.rcParams.update({'font.size': 13})

# ── SLO thresholds ────────────────────────────────────────────────────────────

CHATBOT_TTFT_SLO = 1.0    # seconds
CHATBOT_TPOT_SLO = 0.02   # seconds
DEEP_RESEARCH_SLO = 600.0  # seconds

# ── Colors ────────────────────────────────────────────────────────────────────

COLOR_MET = '#70AD47'
COLOR_MISSED = '#E74C3C'


# ── App-type processors ───────────────────────────────────────────────────────

def process_chatbot(results_list):
    """
    Returns (sub_slos, overall_pct).

    sub_slos: list of dicts {label, pct_met, mean_val, threshold}
    overall_pct: average of all sub-SLO percentages
    """
    requests = [r for r in results_list if r.get('status') == 'chatbot_complete']
    if not requests:
        return [], 0.0

    ttft_vals = [r['ttft'] for r in requests]
    tpot_vals = [r['tpot'] for r in requests]

    ttft_pct = sum(v <= CHATBOT_TTFT_SLO for v in ttft_vals) / len(ttft_vals) * 100
    tpot_pct = sum(v <= CHATBOT_TPOT_SLO for v in tpot_vals) / len(tpot_vals) * 100

    sub_slos = [
        {'label': 'TTFT', 'pct_met': ttft_pct, 'mean_val': np.mean(ttft_vals), 'threshold': CHATBOT_TTFT_SLO},
        {'label': 'TPOT', 'pct_met': tpot_pct, 'mean_val': np.mean(tpot_vals), 'threshold': CHATBOT_TPOT_SLO},
    ]
    overall_pct = np.mean([ttft_pct, tpot_pct])
    return sub_slos, overall_pct


def process_deep_research(results_list):
    """Returns (sub_slos, overall_pct)."""
    record = next((r for r in results_list if r.get('status') == 'deep_research_complete'), None)
    if record is None:
        return [], 0.0

    total_time = record['total time']
    pct_met = 100.0 if total_time <= DEEP_RESEARCH_SLO else 0.0

    sub_slos = [
        {'label': 'Total Time', 'pct_met': pct_met, 'mean_val': total_time, 'threshold': DEEP_RESEARCH_SLO},
    ]
    return sub_slos, pct_met


PROCESSORS = {
    'Chatbot': process_chatbot,
    'DeepResearch': process_deep_research,
}


# ── Log parsing ───────────────────────────────────────────────────────────────

def parse_log_file(path):
    """
    Parse a task_*_perf.log file.

    Returns a task dict or None if the file should be skipped (dummy or unknown
    app type or parse failure).
    """
    try:
        with open(path, 'r') as f:
            content = f.read()
    except OSError as e:
        print(f"Warning: could not read '{path}': {e}")
        return None

    app_type_m = re.search(r'^app_type:\s*(\S+)', content, re.MULTILINE)
    if not app_type_m:
        print(f"Warning: no app_type found in '{path}', skipping.")
        return None

    app_type = app_type_m.group(1)
    if app_type.lower() == 'dummy':
        return None

    if app_type not in PROCESSORS:
        print(f"Warning: unknown app_type '{app_type}' in '{path}', skipping.")
        return None

    task_id_m = re.search(r'^task_id:\s*(\S+)', content, re.MULTILINE)
    start_m = re.search(r'^start_time:\s*([\d.]+)', content, re.MULTILINE)
    end_m = re.search(r'^end_time:\s*([\d.]+)', content, re.MULTILINE)
    results_m = re.search(r'Task .* results:\s*(\[.*\])', content)

    missing = [name for name, m in [('task_id', task_id_m), ('start_time', start_m),
                                     ('end_time', end_m), ('results', results_m)] if not m]
    if missing:
        print(f"Warning: missing fields {missing} in '{path}', skipping.")
        return None

    try:
        results_list = ast.literal_eval(results_m.group(1))
    except (ValueError, SyntaxError) as e:
        print(f"Warning: could not parse results list in '{path}': {e}")
        return None

    sub_slos, overall_pct = PROCESSORS[app_type](results_list)

    return {
        'task_id': task_id_m.group(1),
        'app_type': app_type,
        'start_time': float(start_m.group(1)),
        'end_time': float(end_m.group(1)),
        'sub_slos': sub_slos,
        'overall_pct': overall_pct,
    }


def discover_task_logs(results_dir):
    """Return parsed task dicts for all non-dummy task_*_perf.log files."""
    pattern = os.path.join(results_dir, 'task_*_perf.log')
    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"Error: no task_*_perf.log files found in '{results_dir}'.")
        sys.exit(1)

    tasks = []
    for path in paths:
        task = parse_log_file(path)
        if task is not None:
            tasks.append(task)
    return tasks


# ── Chart ─────────────────────────────────────────────────────────────────────

# Marker shapes cycling for sub-SLOs (up-triangle alternates with down-triangle)
_MARKER_SHAPES = ['^', 'v', 'D', 's']
# Vertical offsets relative to bar center so markers don't overlap the bar
_MARKER_OFFSETS = [0.42, -0.42, 0.55, -0.55]


def create_gantt_chart(tasks, output_path):
    tasks = sorted(tasks, key=lambda t: t['start_time'])

    n = len(tasks)
    fig_height = max(3.5, n * 1.2 + 1.2)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    start_times = [t['start_time'] for t in tasks]
    end_times = [t['end_time'] for t in tasks]
    time_range = max(end_times) - min(start_times)

    # Extra right margin for the annotation text
    x_min = min(start_times) - max(time_range * 0.02, 2)
    x_max = max(end_times) + time_range * 0.35

    legend_handles = []
    legend_done = False

    for i, task in enumerate(tasks):
        start = task['start_time']
        end = task['end_time']
        duration = end - start
        overall_pct = task['overall_pct']

        met_width = duration * overall_pct / 100
        missed_width = duration - met_width

        bar_met = ax.barh(i, met_width, left=start, height=0.6,
                          color=COLOR_MET, alpha=0.9, edgecolor='black', linewidth=0.8)
        bar_missed = ax.barh(i, missed_width, left=start + met_width, height=0.6,
                             color=COLOR_MISSED, alpha=0.9, edgecolor='black', linewidth=0.8)

        if not legend_done:
            legend_handles = [bar_met, bar_missed]
            legend_done = True

        # Sub-SLO markers
        for j, slo in enumerate(task['sub_slos']):
            marker_x = start + (slo['pct_met'] / 100) * duration
            marker_y = i + _MARKER_OFFSETS[j % len(_MARKER_OFFSETS)]
            marker_color = COLOR_MET if slo['pct_met'] >= 95 else COLOR_MISSED
            marker_shape = _MARKER_SHAPES[j % len(_MARKER_SHAPES)]

            ax.plot(marker_x, marker_y, marker=marker_shape, markersize=9,
                    color=marker_color, markeredgecolor='black', markeredgewidth=0.6,
                    linestyle='none', zorder=5)

            label_text = f"{slo['label']}: {slo['pct_met']:.0f}%  (avg {slo['mean_val']:.3f}s, SLO ≤{slo['threshold']}s)"
            text_x = marker_x + time_range * 0.01
            ax.text(text_x, marker_y, label_text,
                    va='center', ha='left', fontsize=9,
                    color=marker_color)

        # Overall annotation to the right of the bar
        overall_color = COLOR_MET if overall_pct >= 95 else COLOR_MISSED
        ax.text(end + time_range * 0.01, i, f'Overall: {overall_pct:.1f}%',
                va='center', ha='left', fontsize=11, fontweight='bold',
                color=overall_color)

    ax.set_yticks(range(n))
    ax.set_yticklabels([t['task_id'] for t in tasks])
    ax.set_xlabel('Time (seconds)')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.8, n - 0.2)
    ax.grid(axis='x', linestyle='--', alpha=0.6)

    # Tick spacing
    if time_range > 1000:
        major_tick, minor_tick = 200, 50
    elif time_range > 400:
        major_tick, minor_tick = 100, 20
    elif time_range > 200:
        major_tick, minor_tick = 50, 10
    elif time_range > 100:
        major_tick, minor_tick = 20, 5
    elif time_range > 50:
        major_tick, minor_tick = 10, 2
    else:
        major_tick, minor_tick = 5, 1

    ax.xaxis.set_major_locator(MultipleLocator(major_tick))
    ax.xaxis.set_minor_locator(MultipleLocator(minor_tick))

    ax.legend(legend_handles, ['SLO Met', 'SLO Missed'],
              loc='upper left', frameon=True, fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, format='png', bbox_inches='tight', dpi=300)
    print(f"Saved: {output_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python gantt_chart.py <results_dir> [output.png]")
        sys.exit(1)

    results_dir = sys.argv[1]
    if not os.path.isdir(results_dir):
        print(f"Error: '{results_dir}' is not a directory.")
        sys.exit(1)

    output_path = (sys.argv[2] if len(sys.argv) > 2
                   else os.path.join(results_dir, 'gantt_slo.png'))

    tasks = discover_task_logs(results_dir)
    if not tasks:
        print("No processable tasks found.")
        sys.exit(1)

    create_gantt_chart(tasks, output_path)


if __name__ == '__main__':
    main()
