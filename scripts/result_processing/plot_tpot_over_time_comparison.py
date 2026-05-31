import argparse
import ast
import json
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
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


def _load_results(file_path):
    with open(file_path, 'r', encoding='utf-8') as handle:
        content = handle.read()

    if not re.search(r"app_type: Chatbot", content):
        raise ValueError(f"Expected app_type: Chatbot in {file_path}")

    match = re.search(r"Task .* results:\s*(.*)", content)
    if not match:
        raise ValueError(f"Could not find task results in {file_path}")

    return ast.literal_eval(match.group(1))[1:-1]


def _read_deep_slo(results_dir):
    config_path = os.path.join(results_dir, 'config.json')
    try:
        with open(config_path, 'r', encoding='utf-8') as handle:
            config = json.load(handle)
        return config.get('deep1', {}).get('tgs_slo')
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return None


def create_comparison_plot(
    chatbot_metrics,
    deep_metrics,
    output_path,
    chatbot_slo,
    deep_slo,
    log=False,
    include_request_markers=False,
):
    chat_tpot_values = [m['tpot'] for m in chatbot_metrics if m.get('tpot') is not None]
    deep_tpot_values = [m['tpot'] for m in deep_metrics if m.get('tpot') is not None]

    chat_x, chat_y, chat_sep = _build_timeline(chatbot_metrics)
    deep_x, deep_y, deep_sep = _build_timeline(deep_metrics)

    if not chat_x:
        raise ValueError('No TPOT samples found in chatbot log')
    if not deep_x:
        raise ValueError('No TPOT samples found in deep research log')

    chat_color = '#1E88E5'
    deep_color = '#D81B60'

    sns.set_style('darkgrid')
    plt.rcParams['figure.figsize'] = (14, 8)
    plt.rcParams['font.family'] = 'DejaVu Sans'

    fig, ax = plt.subplots(1, 1)
    fig.patch.set_facecolor('#F5F5F5')

    ax.plot(chat_x, chat_y, color=chat_color, linewidth=2.0, marker='o', markersize=3.5, label='Chatbot TPOT')
    ax.plot(deep_x, deep_y, color=deep_color, linewidth=2.0, marker='o', markersize=3.5, label='Deep Research TPOT')

    if chatbot_slo is not None:
        ax.axhline(y=chatbot_slo, color=chat_color, linestyle='--', linewidth=2.0,
                   label=f'Chatbot SLO: {chatbot_slo}s')

    if deep_slo is not None:
        ax.axhline(y=deep_slo, color=deep_color, linestyle='--', linewidth=2.0,
                   label=f'Deep Research SLO: {deep_slo}s')

    if include_request_markers:
        for sep in chat_sep:
            ax.axvline(sep, color=chat_color, linestyle='--', linewidth=1.5, alpha=0.4)
        for sep in deep_sep:
            ax.axvline(sep, color=deep_color, linestyle='--', linewidth=1.5, alpha=0.4)

    if log:
        ax.set_yscale('log')
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=10))
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(0.1, 1, 0.1), numticks=20))

    all_y = chat_y + deep_y
    all_x = chat_x + deep_x
    ax.set_ylim(0.01, max(max(all_y) * 1.2, 0.4))
    ax.set_xlim(min(all_x) - 0.1, max(all_x) + 0.1)

    ax.set_xlabel('Elapsed time since first token within each request, stacked across requests (s)', fontsize=13)
    ax.set_ylabel('Seconds', fontsize=13)
    ax.tick_params(axis='both', labelsize=11)
    ax.legend(fontsize=11, loc='upper right')

    def _stats_line(label, tpot_values, slo):
        line = f"{label} — Avg: {np.mean(tpot_values):.3f}s, Max: {max(tpot_values):.3f}s"
        if slo is not None and tpot_values:
            compliance = sum(1 for v in tpot_values if v <= slo) / len(tpot_values) * 100
            line += f", SLO Compliance: {compliance:.1f}%"
        return line

    summary_text = (
        f"Summary Statistics:\n"
        f"{_stats_line('Chatbot', chat_tpot_values, chatbot_slo)}\n"
        f"{_stats_line('Deep Research', deep_tpot_values, deep_slo)}"
    )

    fig.text(0.5, 0.01, summary_text, ha='center', fontsize=11,
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

    plt.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.18)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'Successfully created TPOT comparison plot: {output_path}')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Plot TPOT over time for chatbot and deep research running concurrently.'
    )
    parser.add_argument('results_dir', help='Directory containing task_chat1_u0_perf.log and task_deep1_u0_perf.log')
    parser.add_argument('--no-log', action='store_true', help='Use linear scale for y-axis (default is log scale)')
    parser.add_argument('--chatbot-slo', type=float, default=0.02, metavar='SECONDS',
                        help='Chatbot SLO threshold in seconds (default: 0.02)')
    parser.add_argument('--deep-research-slo', type=float, default=None, metavar='SECONDS',
                        help='Deep research SLO threshold in seconds (default: read from config.json)')
    parser.add_argument('--include-request-markers', action='store_true',
                        help='Draw color-coded vertical lines at request boundaries')
    args = parser.parse_args()

    chat_log = os.path.join(args.results_dir, 'task_chat1_u0_perf.log')
    deep_log = os.path.join(args.results_dir, 'task_deep1_u0_perf.log')

    for path in (chat_log, deep_log):
        if not os.path.exists(path):
            print(f'Error: {path} not found')
            sys.exit(1)

    chatbot_metrics = _load_results(chat_log)
    deep_metrics = _load_results(deep_log)

    deep_slo = args.deep_research_slo
    if deep_slo is None:
        deep_slo = _read_deep_slo(args.results_dir)

    output_path = os.path.join(args.results_dir, 'tpot_comparison.png')
    create_comparison_plot(
        chatbot_metrics,
        deep_metrics,
        output_path,
        chatbot_slo=args.chatbot_slo,
        deep_slo=deep_slo,
        log=not args.no_log,
        include_request_markers=args.include_request_markers,
    )


if __name__ == '__main__':
    main()
