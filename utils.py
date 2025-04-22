import numpy as np
import matplotlib.pyplot as plt
import os
import torch

def create_results_directory(algorithm_name):
    """Create directory for saving results if it doesn't exist"""
    results_dir = f"results/{algorithm_name}"
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

def save_model(model, algorithm_name):
    """Save the model weights"""
    results_dir = create_results_directory(algorithm_name)
    torch.save(model.state_dict(), f"{results_dir}/model.pt")

def plot_learning_curve(rewards, algorithm_name, window=100):
    """Plot and save the episode rewards learning curve"""
    results_dir = create_results_directory(algorithm_name)
    
    # Calculate moving average if there are enough points
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
    else:
        moving_avg = rewards
    
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, alpha=0.3)
    plt.plot(np.arange(len(moving_avg)), moving_avg)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title(f'Learning Curve - {algorithm_name}')
    
    # Save the figure
    plt.savefig(f"{results_dir}/learning_curve.png")
    plt.close()

def plot_metrics(algorithm_name, rewards, moisture_levels, actions, optimal_lower=60, optimal_upper=80):
    """Plot and save comprehensive metrics for algorithm performance"""
    results_dir = create_results_directory(algorithm_name)
    
    fig, axs = plt.subplots(3, 1, figsize=(12, 15))
    
    # Plot rewards
    axs[0].plot(rewards)
    axs[0].set_xlabel('Episode')
    axs[0].set_ylabel('Total Reward')
    axs[0].set_title(f'{algorithm_name} - Episode Rewards')
    
    # Plot final episode moisture levels
    if moisture_levels:
        axs[1].plot(moisture_levels[-1])
        axs[1].axhline(y=optimal_lower, color='g', linestyle='--', label='Optimal Lower')
        axs[1].axhline(y=optimal_upper, color='g', linestyle='--', label='Optimal Upper')
        axs[1].set_xlabel('Step')
        axs[1].set_ylabel('Soil Moisture (%)')
        axs[1].set_title('Final Episode - Soil Moisture Levels')
        axs[1].legend()
    
    # Plot final episode actions
    if actions:
        axs[2].step(range(len(actions[-1])), actions[-1])
        axs[2].set_xlabel('Step')
        axs[2].set_ylabel('Irrigation Action')
        axs[2].set_title('Final Episode - Irrigation Actions')
        axs[2].set_yticks([0, 1, 2, 3])
        axs[2].set_yticklabels(['None', 'Low', 'Medium', 'High'])
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/performance_metrics.png")
    plt.close()

def plot_comparison(algorithms, avg_rewards, std_rewards, moisture_in_range_pcts, water_usage):
    """Plot and save comparison between different algorithms"""
    results_dir = "results/comparison"
    os.makedirs(results_dir, exist_ok=True)
    
    # Create subplots
    fig, axs = plt.subplots(3, 1, figsize=(12, 15))
    
    # Plot average rewards
    x = np.arange(len(algorithms))
    axs[0].bar(x, avg_rewards, yerr=std_rewards, capsize=10)
    axs[0].set_ylabel('Avg Reward (last 100 episodes)')
    axs[0].set_title('Algorithm Performance Comparison')
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(algorithms)
    
    # Plot moisture in optimal range percentage
    axs[1].bar(x, moisture_in_range_pcts)
    axs[1].set_ylabel('% Time Moisture in Optimal Range')
    axs[1].set_title('Irrigation Effectiveness')
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(algorithms)
    
    # Plot water usage
    axs[2].bar(x, water_usage)
    axs[2].set_ylabel('Average Water Usage')
    axs[2].set_title('Water Efficiency')
    axs[2].set_xticks(x)
    axs[2].set_xticklabels(algorithms)
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/algorithm_comparison.png")
    plt.close()

def evaluate_policy(env, policy, episodes=10):
    """
    Evaluate a policy over several episodes
    
    Args:
        env: The environment
        policy: A callable that takes a state and returns an action
        episodes: Number of episodes to evaluate
    
    Returns:
        avg_reward: Average total reward across episodes
        success_rate: Percentage of time moisture level was in optimal range
        avg_water_usage: Average water usage per episode
    """
    total_rewards = []
    optimal_time_steps = 0
    total_time_steps = 0
    total_water_usage = 0
    
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        episode_steps = 0
        
        while not done:
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # Track metrics
            episode_reward += reward
            episode_steps += 1
            
            # Check if moisture is in optimal range
            soil_moisture = next_state[0]
            if env.optimal_moisture_lower <= soil_moisture <= env.optimal_moisture_upper:
                optimal_time_steps += 1
            
            # Track water usage
            if action == 0:
                water_amount = 0
            elif action == 1:
                water_amount = 5
            elif action == 2:
                water_amount = 15
            else:
                water_amount = 25
            total_water_usage += water_amount
            
            state = next_state
            done = terminated or truncated
        
        total_rewards.append(episode_reward)
        total_time_steps += episode_steps
    
    avg_reward = np.mean(total_rewards)
    success_rate = optimal_time_steps / total_time_steps * 100 if total_time_steps > 0 else 0
    avg_water_usage = total_water_usage / episodes
    
    return avg_reward, success_rate, avg_water_usage 