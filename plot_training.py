import matplotlib.pyplot as plt
import json
import os
import numpy as np
import config

def smooth(data, window=50):
    """Simple moving average smoothing."""
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode='valid')


def get_series(data, *names):
    for name in names:
        values = data.get(name, [])
        if values:
            return values
    return []

def plot_training_results():
    log_file = config.LOG_FILE
    
    if not os.path.exists(log_file):
        print(f"No log file found at {log_file}")
        return

    with open(log_file, 'r') as f:
        data = json.load(f)

    episodes = data['episodes']
    scores = data['scores']
    rewards = get_series(data, 'rewards')
    losses = get_series(data, 'losses', 'play_losses')
    value_losses = get_series(data, 'value_losses', 'play_value_losses')
    entropies = get_series(data, 'entropies', 'play_entropies')
    clip_fractions = get_series(data, 'clip_fractions', 'play_clip_fractions')
    explained_variance = get_series(data, 'explained_variance', 'play_explained_variance')

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Hearts AI Training Progress', fontsize=14, fontweight='bold')

    # 1. Average Score (Lower is Better) - THE KEY METRIC
    ax1 = axes[0, 0]
    ax1.plot(episodes, scores, alpha=0.3, color='blue', label='Raw')
    if len(scores) > 50:
        smoothed = smooth(scores, 50)
        ax1.plot(episodes[49:], smoothed, color='blue', linewidth=2, label='Smoothed')
    ax1.axhline(y=6.5, color='green', linestyle='--', alpha=0.7, label='Random Baseline (~6.5)')
    ax1.axhline(y=13, color='red', linestyle='--', alpha=0.7, label='Poor Performance (13)')
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Score')
    ax1.set_title('Average Score (Lower is Better) ⬇️')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # 2. Value Loss (Should decrease over time)
    ax2 = axes[0, 1]
    if value_losses:
        ax2.plot(episodes, value_losses, alpha=0.3, color='orange', label='Raw')
        if len(value_losses) > 50:
            smoothed = smooth(value_losses, 50)
            ax2.plot(episodes[49:], smoothed, color='orange', linewidth=2, label='Smoothed')
        ax2.set_xlabel('Episodes')
        ax2.set_ylabel('Value Loss')
        ax2.set_title('Value Function Loss (Should Decrease) ⬇️')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No Value Loss Data', ha='center', va='center', transform=ax2.transAxes)

    # 3. Average Reward (Should increase over time)
    ax3 = axes[1, 0]
    if rewards:
        ax3.plot(episodes, rewards, alpha=0.3, color='green', label='Raw')
        if len(rewards) > 50:
            smoothed = smooth(rewards, 50)
            ax3.plot(episodes[49:], smoothed, color='green', linewidth=2, label='Smoothed')
        ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Episodes')
        ax3.set_ylabel('Reward')
        ax3.set_title('Average Episode Reward (Higher is Better) ⬆️')
        ax3.legend(loc='lower right')
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No Reward Data', ha='center', va='center', transform=ax3.transAxes)

    # 4. Entropy and PPO diagnostics
    ax4 = axes[1, 1]
    if entropies:
        ax4.plot(episodes, entropies, alpha=0.3, color='purple', label='Raw')
        if len(entropies) > 50:
            smoothed = smooth(entropies, 50)
            ax4.plot(episodes[49:], smoothed, color='purple', linewidth=2, label='Smoothed')
        ax4.set_xlabel('Episodes')
        ax4.set_ylabel('Entropy')
        ax4.set_title('Policy Entropy and PPO Diagnostics')
        ax4.legend(loc='upper right')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No Entropy Data', ha='center', va='center', transform=ax4.transAxes)

    if clip_fractions:
        ax4b = ax4.twinx()
        ax4b.plot(episodes, clip_fractions, alpha=0.4, color='red', label='Clip Fraction')
        ax4b.set_ylabel('Clip Fraction')
        ax4b.legend(loc='lower right')

    if explained_variance:
        print(f"Latest explained variance: {explained_variance[-1]:.3f}")
    if losses:
        print(f"Latest total loss: {losses[-1]:.3f}")

    plt.tight_layout()
    save_path = os.path.join(config.OUTPUT_DIR, 'training_plot.png')
    plt.savefig(save_path, dpi=150)
    print(f"Plot saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    plot_training_results()
