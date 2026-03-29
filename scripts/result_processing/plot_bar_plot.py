import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import parse_results_chatbot_log as chat
import parse_results_lv_log as lv
import parse_results_imagegen_log as img

# Global font size
plt.rcParams.update({'font.size': 14})

SLOs = {
    'chatbot-ttft': 1,
    'chatbot-tpot': 0.25,
    'imagegen': 28,
    'livecaption': 2
}

def plot_performance(gpu_folder_path, cpu_folder_path, save_path="scripts/plots/gpu_vs_cpu_latency_and_slo_sampling"):
    # Read CSVs
    # gpu_data = pd.read_csv(os.path.join(gpu_folder_path, 'task_chat1_u0_perf.csv'))
    # cpu_data = pd.read_csv(os.path.join(cpu_folder_path, 'task_chat1_u0_perf.csv'))
    chat_bot_gpu_data = pd.read_csv(os.path.join(gpu_folder_path, 'task_chat1_u0_perf.csv'))
    chat_bot_cpu_data = pd.read_csv(os.path.join(cpu_folder_path, 'task_chat1_u0_perf.csv'))
    image_gen_gpu_data = pd.read_csv(os.path.join(gpu_folder_path, 'task_imagegen1_u0_perf.csv'))
    image_gen_cpu_data = pd.read_csv(os.path.join(cpu_folder_path, 'task_imagegen1_u0_perf.csv'))
    live_caption_gpu_data = pd.read_csv(os.path.join(gpu_folder_path, 'task_lv1_u0_perf.csv'))
    live_caption_cpu_data = pd.read_csv(os.path.join(cpu_folder_path, 'task_lv1_u0_perf.csv'))

    # print min max
    print(image_gen_gpu_data['total time'].min(), image_gen_gpu_data['total time'].max())
    print(live_caption_gpu_data['time'].min(), live_caption_gpu_data['time'].max())

    chatbot_ttft_gpu = chat_bot_gpu_data['ttft'].mean() / SLOs['chatbot-ttft']
    chatbot_tpot_gpu = chat_bot_gpu_data['tpot'].mean() / SLOs['chatbot-tpot']
    chatbot_slo_gpu = 100 * (1 - ((chat_bot_gpu_data['ttft'] > SLOs['chatbot-ttft']) |
                                  (chat_bot_gpu_data['tpot'] > SLOs['chatbot-tpot'])).sum() / len(chat_bot_gpu_data))
    chatbot_ttft_cpu = chat_bot_cpu_data['ttft'].mean() / SLOs['chatbot-ttft']
    chatbot_tpot_cpu = chat_bot_cpu_data['tpot'].mean() / SLOs['chatbot-tpot']
    chatbot_slo_cpu = 100 * (1 - ((chat_bot_cpu_data['ttft'] > SLOs['chatbot-ttft']) |
                                  (chat_bot_cpu_data['tpot'] > SLOs['chatbot-tpot'])).sum() / len(chat_bot_cpu_data))
    imagegen_latency_gpu = image_gen_gpu_data['total time'].mean() / SLOs['imagegen']
    imagegen_slo_gpu = 100 * (1 - (image_gen_gpu_data['total time'] > SLOs['imagegen']).sum() / len(image_gen_gpu_data))
    imagegen_latency_cpu = image_gen_cpu_data['total time'].mean() / SLOs['imagegen']
    imagegen_slo_cpu = 100 * (1 - (image_gen_cpu_data['total time'] > SLOs['imagegen']).sum() / len(image_gen_cpu_data))
    livecaption_latency_gpu = live_caption_gpu_data['time'].mean() / SLOs['livecaption']
    livecaption_slo_gpu = 100 * (1 - (live_caption_gpu_data['time'] > SLOs['livecaption']).sum() / len(live_caption_gpu_data))
    livecaption_latency_cpu = live_caption_cpu_data['time'].mean() / SLOs['livecaption']
    livecaption_slo_cpu = 100 * (1 - (live_caption_cpu_data['time'] > SLOs['livecaption']).sum() / len(live_caption_cpu_data))

    gpu_latency_values = [chatbot_ttft_gpu, chatbot_tpot_gpu, imagegen_latency_gpu, livecaption_latency_gpu]
    gpu_latency_stds = [chat_bot_gpu_data['ttft'].std() / SLOs['chatbot-ttft'],
                        chat_bot_gpu_data['tpot'].std() / SLOs['chatbot-tpot'],
                        image_gen_gpu_data['total time'].std() / SLOs['imagegen'],
                        live_caption_gpu_data['time'].std() / SLOs['livecaption']]
    latency_colors = ['#778899', '#778899', '#A0522D', '#C71585']
    gpu_latency_hatch = '//'

    cpu_latency_values = [chatbot_ttft_cpu, chatbot_tpot_cpu, imagegen_latency_cpu, livecaption_latency_cpu]
    cpu_latency_stds = [chat_bot_cpu_data['ttft'].std() / SLOs['chatbot-ttft'],
                        chat_bot_cpu_data['tpot'].std() / SLOs['chatbot-tpot'],
                        image_gen_cpu_data['total time'].std() / SLOs['imagegen'],
                        live_caption_cpu_data['time'].std() / SLOs['livecaption']]

    gpu_slo_values = [chatbot_slo_gpu, imagegen_slo_gpu, livecaption_slo_gpu]
    gpu_slo_colors = ['#778899', '#A0522D', '#C71585']
    cpu_slo_values = [chatbot_slo_cpu, imagegen_slo_cpu, livecaption_slo_cpu]
    cpu_slo_colors = ['#778899', '#A0522D', '#C71585']

    # X positions
    latency_x = [0, 1, 2, 3, 5, 6, 7, 8]
    alpha = 1
    bar_width = 0.8
    edge_width = 1
    fig, ax = plt.subplots(figsize=(12, 6))

    latency_values =  gpu_latency_values + cpu_latency_values
    latency_stds = gpu_latency_stds + cpu_latency_stds
    latency_colors *= 2

    ax.bar(latency_x, latency_values, yerr=latency_stds, width=bar_width,
           color=latency_colors, hatch=gpu_latency_hatch, alpha=alpha,
           edgecolor=latency_colors, linewidth=edge_width, facecolor='white', log=True)
    ax.set_yscale('log')
    ax.set_ylabel('Normalized Latency (log scale)', color='black')
    # ax.set_ylim(1e-2, max(latency_values) * 1.8)
    ax.set_ylim(1e-3, 1e3 + 1e3)
    ax.set_xlim(-1, 9)

    # X-axis ticks just at group centers
    ax.set_xticks([1.5, 6.5])
    ax.set_xticklabels(['GPU', 'CPU'])

    for i in [0, 1, 5, 6]:
        label = 'TTFT' if i == 0 or i == 5 else 'TPOT'
        ax.text(i, 1e-3, label,
                ha='center', va='bottom')

    height = 1.5
    ax.text(0.8, height, 'Latency Threshold',
            ha='center', va='bottom', color='green')
    ax.hlines(y=1.0, xmin=-1, xmax=9, color='green', linestyle='--', linewidth=2)

    # Sorted labels and handles
    legend_labels = [
        'Chatbot',
        'ImageGen',
        'LiveCaptions',
    ]

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor=latency_colors[0], hatch=gpu_latency_hatch, linewidth=edge_width),
        plt.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor=latency_colors[2], hatch=gpu_latency_hatch, linewidth=edge_width),
        plt.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor=latency_colors[3], hatch=gpu_latency_hatch, linewidth=edge_width),
    ]

    # Display in two rows of 4 items max
    ax.legend(legend_handles, legend_labels, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.2))


    fig.tight_layout()
    save_path_lat = os.path.join(save_path, 'gpu_vs_cpu_latency.pdf')
    plt.savefig(save_path_lat)
    print(f"Latency saved to {save_path_lat}")

    # clear
    plt.clf()
    plt.close()

    # SLO bars
    fig, ax2 = plt.subplots(figsize=(12, 6))
    slo_x = [1, 2, 3, 5, 6, 7]
    slo_values = gpu_slo_values + cpu_slo_values
    slo_colors = gpu_slo_colors + cpu_slo_colors
    slo_hatch = 'x'

    # Annotate all SLO percentages
    for idx, val in enumerate(slo_values):
        ax2.text(slo_x[idx], val + 2, f'{int(val)}%', ha='center', va='bottom', color=slo_colors[idx])


    ax2.bar(slo_x, slo_values, width=bar_width, color=slo_colors, hatch=slo_hatch,
            alpha=alpha, edgecolor=slo_colors, linewidth=edge_width, facecolor='white')
    ax2.set_ylabel('SLO Attainment (%)', color='black')
    ax2.set_ylim(0, 115)

    # X-axis ticks just at group centers
    ax2.set_xticks([2, 6])
    ax2.set_xticklabels(['GPU', 'CPU'])
    ax2.set_yticks([0, 20, 40, 60, 80, 100])
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.set_xlim(-1, 9)

    height = 102
    # ax2.text(6, height, 'SLO Threshold',
            # ha='center', va='bottom', fontsize=16, color='green')
    ax2.hlines(y=100.0, xmin=-1, xmax=9, color='green', linestyle='--', linewidth=2)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor=slo_colors[0], hatch=slo_hatch, linewidth=edge_width),
        plt.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor=slo_colors[1], hatch=slo_hatch, linewidth=edge_width),
        plt.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor=slo_colors[2], hatch=slo_hatch, linewidth=edge_width),
    ]

    ax2.legend(legend_handles, legend_labels, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.2))


    fig.tight_layout()
    save_path = os.path.join(save_path, 'gpu_vs_cpu_slo_sampling.pdf')
    plt.savefig(save_path)
    print(f"Latency saved to {save_path}")




def plot_performance_bar_plots(folder_path):
    # Generate CSVs
    chat.parse_results_from_file(os.path.join(folder_path, 'task_chat1_u0_perf.log'))
    img.parse_results_from_file(os.path.join(folder_path, 'task_imagegen1_u0_perf.log'))
    lv.parse_results_from_file(os.path.join(folder_path, 'task_lv1_u0_perf.log')) 

    # Read CSVs
    chatbot_data = pd.read_csv(os.path.join(folder_path, 'task_chat1_u0_perf.csv'))
    imagegen_data = pd.read_csv(os.path.join(folder_path, 'task_imagegen1_u0_perf.csv'))
    livecaption_data = pd.read_csv(os.path.join(folder_path, 'task_lv1_u0_perf.csv'))

    # Metrics
    chatbot_ttft = chatbot_data['ttft'].mean() / SLOs['chatbot-ttft']
    chatbot_tpot = chatbot_data['tpot'].mean() / SLOs['chatbot-tpot']
    chatbot_slo = 100 * (1 - ((chatbot_data['ttft'] > SLOs['chatbot-ttft']) |
                              (chatbot_data['tpot'] > SLOs['chatbot-tpot'])).sum() / len(chatbot_data))
    chatbot_ttft_std = chatbot_data['ttft'].std() / SLOs['chatbot-ttft']
    chatbot_tpot_std = chatbot_data['tpot'].std() / SLOs['chatbot-tpot']

    imagegen_latency = imagegen_data['total time'].mean() / SLOs['imagegen']
    imagegen_slo = 100 * (1 - (imagegen_data['total time'] > SLOs['imagegen']).sum() / len(imagegen_data))
    imagegen_latency_std = imagegen_data['total time'].std() / SLOs['imagegen']

    livecaption_latency = livecaption_data['time'].mean() / SLOs['livecaption']
    livecaption_slo = 100 * (1 - (livecaption_data['time'] > SLOs['livecaption']).sum() / len(livecaption_data))
    livecaption_latency_std = livecaption_data['time'].std() / SLOs['livecaption']

    # Bar data
    latency_values = [chatbot_ttft, chatbot_tpot, imagegen_latency, livecaption_latency]
    latency_stds = [chatbot_ttft_std, chatbot_tpot_std, imagegen_latency_std, livecaption_latency_std]
    latency_colors = ['#778899', '#778899', '#A0522D', '#C71585']
    latency_hatch = '+'

    slo_values = [chatbot_slo, imagegen_slo, livecaption_slo]
    slo_colors = ['#778899', '#A0522D', '#C71585']
    slo_hatch = 'x'

    # Plot in two panels to avoid clutter from mixing two different y-axes.
    bar_width = 0.7
    alpha = 1
    edge_width = 1.8
    fig, (ax_lat, ax_slo) = plt.subplots(
        1,
        2,
        figsize=(15, 7.5),
        gridspec_kw={'width_ratios': [1.35, 1]}
    )

    # Latency panel
    latency_x = np.arange(len(latency_values))
    latency_labels = ['Chatbot\nTTFT', 'Chatbot\nTPOT', 'ImageGen', 'LiveCaption']
    ax_lat.bar(
        latency_x,
        latency_values,
        yerr=latency_stds,
        width=bar_width,
        color=latency_colors,
        hatch=latency_hatch,
        alpha=alpha,
        edgecolor=latency_colors,
        linewidth=edge_width,
        facecolor='white'
    )
    ax_lat.set_yscale('log')
    # Log scale cannot include negative values; this sets 10^-2 to 10^2.
    ax_lat.set_ylim(1e-2, 1e2)
    ax_lat.set_xticks(latency_x)
    ax_lat.set_xticklabels(latency_labels)
    ax_lat.set_ylabel('Normalized Latency (log scale)')
    ax_lat.set_title('Latency')
    ax_lat.axhline(y=1.0, color='green', linestyle='--', linewidth=2, label='Latency Threshold')
    ax_lat.grid(axis='y', which='both', alpha=0.2)

    # SLO panel
    slo_x = np.arange(len(slo_values))
    slo_labels = ['Chatbot', 'ImageGen', 'LiveCaption']
    ax_slo.bar(
        slo_x,
        slo_values,
        width=bar_width,
        color=slo_colors,
        hatch=slo_hatch,
        alpha=alpha,
        edgecolor=slo_colors,
        linewidth=edge_width,
        facecolor='white'
    )
    ax_slo.set_ylim(0, 110)
    ax_slo.set_yticks([0, 20, 40, 60, 80, 100])
    ax_slo.set_xticks(slo_x)
    ax_slo.set_xticklabels(slo_labels)
    ax_slo.set_ylabel('SLO Attainment (%)')
    ax_slo.set_title('SLO')
    ax_slo.axhline(y=100.0, color='green', linestyle='--', linewidth=2, label='SLO Threshold')
    ax_slo.grid(axis='y', alpha=0.2)

    # Annotate SLO percentages
    for i, val in enumerate(slo_values):
        ax_slo.text(i, val + 1.5, f'{int(val)}%', ha='center', va='bottom', color=slo_colors[i])

    # Shared legend with app colors and threshold line.
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor=latency_colors[0], hatch=latency_hatch, linewidth=edge_width),
        plt.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor=latency_colors[2], hatch=latency_hatch, linewidth=edge_width),
        plt.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor=latency_colors[3], hatch=latency_hatch, linewidth=edge_width),
        plt.Line2D([], [], color='green', linestyle='--', linewidth=2),
    ]
    legend_labels = ['Chatbot', 'ImageGen', 'LiveCaption', 'Threshold']
    fig.legend(
        legend_handles,
        legend_labels,
        loc='upper center',
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.985)
    )

    # Layout and save
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.87])
    plot_path = os.path.join(folder_path, 'performance_split_barplot_final_labeled.pdf')
    plt.savefig(plot_path, bbox_inches='tight', pad_inches=0.25)
    plt.close(fig)
    print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    # Check if a file path was provided as command line argument
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
        plot_performance_bar_plots(folder_path)
    else:
        print("Usage: python script_name.py <path_to_results_file>")
        print("Example: python parse_results.py log_file.txt")
