import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import time

from environment import SmartIrrigationEnv
from dqn import DQNAgent, train_dqn
from actor_critic import ActorCriticAgent, train_actor_critic
from ppo import PPOAgent, train_ppo
from utils import evaluate_policy, plot_comparison

def main():
    # Create results directory
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/comparison", exist_ok=True)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Training parameters
    num_episodes = 10000  # Reduced for faster comparison
    algorithms = ["DQN", "ActorCritic", "PPO"]
    
    # Results storage
    all_rewards = {}
    all_moisture_levels = {}
    all_actions = {}
    training_times = {}
    
    # Train and evaluate each algorithm
    for algo in algorithms:
        print(f"\n=== Training {algo} ===\n")
        
        # Create environment
        env = SmartIrrigationEnv()
        
        # Train the agent
        start_time = time.time()
        
        if algo == "DQN":
            agent, rewards, moisture_levels, actions = train_dqn(env, num_episodes=num_episodes)
            policy = agent.get_policy()
        elif algo == "ActorCritic":
            agent, rewards, moisture_levels, actions = train_actor_critic(env, num_episodes=num_episodes)
            policy = agent.get_policy()
        elif algo == "PPO":
            agent, rewards, moisture_levels, actions = train_ppo(env, num_episodes=num_episodes)
            policy = agent.get_policy()
        
        training_time = time.time() - start_time
        training_times[algo] = training_time
        
        # Store results
        all_rewards[algo] = rewards
        all_moisture_levels[algo] = moisture_levels
        all_actions[algo] = actions
        
        print(f"\n{algo} training completed in {training_time:.2f} seconds\n")
        
        # Final evaluation
        avg_reward, success_rate, avg_water_usage = evaluate_policy(env, policy, episodes=30)
        
        print(f"{algo} Final Evaluation Results:")
        print(f"Average Reward: {avg_reward:.2f}")
        print(f"Success Rate (moisture in optimal range): {success_rate:.2f}%")
        print(f"Average Water Usage: {avg_water_usage:.2f}")
    
    # Prepare comparison metrics
    avg_rewards = []
    std_rewards = []
    success_rates = []
    water_usages = []
    
    for algo in algorithms:
        # Get last 100 episodes (or all if less than 100)
        last_rewards = all_rewards[algo][-min(100, len(all_rewards[algo])):]
        avg_rewards.append(np.mean(last_rewards))
        std_rewards.append(np.std(last_rewards))
        
        # Re-evaluate for consistent comparison
        env = SmartIrrigationEnv()
        
        if algo == "DQN":
            agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)
            agent.q_network.load_state_dict(torch.load(f"results/{algo}/model.pt"))
            policy = agent.get_policy()
        elif algo == "ActorCritic":
            agent = ActorCriticAgent(env.observation_space.shape[0], env.action_space.n)
            agent.network.load_state_dict(torch.load(f"results/{algo}/model.pt"))
            policy = agent.get_policy()
        elif algo == "PPO":
            agent = PPOAgent(env.observation_space.shape[0], env.action_space.n)
            agent.actor.load_state_dict(torch.load(f"results/{algo}/model.pt"))
            policy = agent.get_policy()
        
        avg_reward, success_rate, avg_water_usage = evaluate_policy(env, policy, episodes=30)
        success_rates.append(success_rate)
        water_usages.append(avg_water_usage)
    
    # Plot comparison
    plot_comparison(algorithms, avg_rewards, std_rewards, success_rates, water_usages)
    
    # Plot learning curves together
    plt.figure(figsize=(12, 8))
    
    for algo in algorithms:
        rewards = all_rewards[algo]
        window = min(100, len(rewards))
        if window > 0:
            smoothed_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
            plt.plot(smoothed_rewards, label=algo)
    
    plt.xlabel('Episode')
    plt.ylabel('Smoothed Reward')
    plt.title('Learning Curves Comparison')
    plt.legend()
    plt.savefig("results/comparison/learning_curves.png")
    plt.close()
    
    # Print training time comparison
    print("\nTraining Time Comparison:")
    for algo in algorithms:
        print(f"{algo}: {training_times[algo]:.2f} seconds")

if __name__ == "__main__":
    main() 