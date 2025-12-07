import json
import matplotlib.pyplot as plt
import config
import os

def plot_training_curves():
    if not os.path.exists(config.LOG_FILE):
        print(f"Log file {config.LOG_FILE} not found.")
        return
        
    with open(config.LOG_FILE, 'r') as f:
        data = json.load(f)
        
    episodes = data['episodes']
    scores = data['scores']
    rewards = data['rewards']
    
    # Calculate Moving Average
    window_size = 50
    def moving_average(data, window):
        if len(data) < window:
            return data
        return [sum(data[i:i+window])/window for i in range(len(data)-window+1)]
    
    avg_scores = moving_average(scores, window_size)
    avg_rewards = moving_average(rewards, window_size)
    
    # Adjust x-axis for moving average
    ma_episodes = episodes[len(episodes)-len(avg_scores):]
    
    plt.figure(figsize=(12, 5))
    
    # Plot Scores (Lower is better)
    plt.subplot(1, 2, 1)
    plt.plot(episodes, scores, alpha=0.3, color='blue', label='Raw Score')
    plt.plot(ma_episodes, avg_scores, color='blue', linewidth=2, label=f'Avg Score ({window_size})')
    plt.xlabel('Episodes')
    plt.ylabel('Game Score (Lower is Better)')
    plt.title('Training Score Progress')
    plt.legend()
    plt.grid(True)
    
    # Plot Rewards (Higher is better)
    plt.subplot(1, 2, 2)
    plt.plot(episodes, rewards, alpha=0.3, color='green', label='Raw Reward')
    plt.plot(ma_episodes, avg_rewards, color='green', linewidth=2, label=f'Avg Reward ({window_size})')
    plt.xlabel('Episodes')
    plt.ylabel('Reward (Higher is Better)')
    plt.title('Training Reward Progress')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    print("Plot saved to training_curves.png")
    plt.show()

if __name__ == "__main__":
    plot_training_curves()
