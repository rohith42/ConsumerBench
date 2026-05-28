import ast
import csv
import glob
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator


def _extract_tpot_samples(metrics):
    raw_samples = metrics.get('tpot_estimates') or []

    if isinstance(raw_samples, dict) and isinstance(raw_samples.get('tpot'), list):
        elapsed_values = raw_samples.get('elapsed_seconds') or [None] * len(raw_samples['tpot'])
        samples = [
            {'elapsed': elapsed_values[index] if index < len(elapsed_values) else None, 'tpot': tpot_value}
            for index, tpot_value in enumerate(raw_samples['tpot'])
            if tpot_value is not None
        ]
    elif isinstance(raw_samples, list) and raw_samples and isinstance(raw_samples[0], dict):
        samples = [
            {
                'elapsed': sample.get('elapsed', sample.get('elapsed_since_first_token')),
                'tpot': sample.get('tpot', sample.get('estimated_tpot')),
            }
            for sample in raw_samples
            if sample.get('tpot', sample.get('estimated_tpot')) is not None
        ]
    else:
        samples = [
            {'elapsed': float(index), 'tpot': tpot_value}
            for index, tpot_value in enumerate(raw_samples)
            if tpot_value is not None
        ]

    if not samples and metrics.get('tpot') is not None:
        samples.append({'elapsed': 0.0, 'tpot': metrics['tpot']})

    return samples


def _load_chatbot_results(file_path):
    with open(file_path, 'r', encoding='utf-8') as handle:
        file_content = handle.read()

    if not re.search(r"app_type: Chatbot", file_content):
        return None

    match = re.search(r"Task .* results:\s*(.*)", file_content)
    if not match:
        raise ValueError(f"Could not find task results in {file_path}")

    results_list = ast.literal_eval(match.group(1))
    return results_list[1:-1]


def _build_timeline(metrics_dicts):
    timeline_x = []
    timeline_y = []
    separators = []
    x_offset = 0.0
    gap = 0.5

    for run_index, metrics in enumerate(metrics_dicts):
        samples = _extract_tpot_samples(metrics)
        if not samples:
            continue

        run_x_values = []
        for sample_index, sample in enumerate(samples):
            elapsed = sample['elapsed']
            if elapsed is None:
                elapsed = float(sample_index)
            run_x_values.append(x_offset + float(elapsed))
            timeline_y.append(sample['tpot'])

        timeline_x.extend(run_x_values)
        if run_index < len(metrics_dicts) - 1:
            separators.append(run_x_values[-1] + gap / 2)
        x_offset = run_x_values[-1] + gap

    return timeline_x, timeline_y, separators


def create_plot(metrics_dicts, output_base, log=False):
    tpot_values = [metrics['tpot'] for metrics in metrics_dicts if metrics.get('tpot') is not None]
    timeline_x, timeline_y, separators = _build_timeline(metrics_dicts)

    if not timeline_x:
        raise ValueError('No TPOT samples found in the chatbot results file')

    tpot_slo = 0.02

    import seaborn as sns

    sns.set_style('darkgrid')
    plt.rcParams['figure.figsize'] = (14, 8)
    plt.rcParams['font.family'] = 'DejaVu Sans'

    fig, ax = plt.subplots(1, 1)
    fig.suptitle('Chatbot TPOT Over Time', fontsize=20, fontweight='bold', y=0.98)
    fig.patch.set_facecolor('#F5F5F5')

    line_color = '#1E88E5'
    slo_line_color = '#FF9800'

    ax.plot(
        timeline_x,
        timeline_y,
        color=line_color,
        linewidth=2.0,
        marker='o',
        markersize=3.5,
        label='TPOT estimates',
    )

    for separator in separators:
        ax.axvline(separator, color=slo_line_color, linestyle='--', linewidth=1.5, alpha=0.5)

    ax.axhline(y=tpot_slo, color=slo_line_color, linestyle='-', linewidth=2.0, label=f'SLO Threshold: {tpot_slo}s')
    ax.fill_between([min(timeline_x), max(timeline_x)], 0, tpot_slo, color=slo_line_color, alpha=0.08)

    if log:
        ax.set_yscale('log')
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=10))
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(0.1, 1, 0.1), numticks=20))
    ax.set_ylim(0.01, max(max(timeline_y) * 1.2, 0.4))

    ax.set_title('TPOT Across Requests and Streaming Progress', fontsize=16, pad=15)
    ax.set_xlabel('Elapsed time since first token within each request, stacked across requests (s)', fontsize=13)
    ax.set_ylabel('Seconds', fontsize=13)
    ax.tick_params(axis='both', labelsize=11)
    ax.set_xlim(min(timeline_x) - 0.1, max(timeline_x) + 0.1)
    ax.legend(fontsize=11, loc='upper right')

    summary_text = (
        f"Summary Statistics:\n"
        f"TPOT - Avg: {np.mean(tpot_values):.3f}s, Max: {max(tpot_values):.3f}s, "
        f"SLO Compliance: {sum(1 for v in tpot_values if v <= tpot_slo) / len(tpot_values) * 100:.1f}%"
    )

    fig.text(0.5, 0.01, summary_text, ha='center', fontsize=11,
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.12)

    output_path = f'{output_base}_tpot_time.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'Successfully created TPOT timeline plot: {output_path}')
    plt.close()


def main():
    if len(sys.argv) < 2:
        print('Usage: python plot_chatbot_tpot_over_time.py <results_dir_or_log_file> [--log]')
        sys.exit(1)

    input_path = sys.argv[1]
    log = '--log' in sys.argv[2:]
    if os.path.isdir(input_path):
        log_files = sorted(glob.glob(os.path.join(input_path, 'task_*_perf.log')))
    else:
        log_files = [input_path]

    if not log_files:
        print(f'No chatbot task logs found in {input_path}')
        sys.exit(1)

    for log_file in log_files:
        metrics_dicts = _load_chatbot_results(log_file)
        if not metrics_dicts:
            continue
        output_base = os.path.splitext(log_file)[0]
        create_plot(metrics_dicts, output_base, log=log)


if __name__ == '__main__':
    main()