import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
import os

from environment import SmartIrrigationEnv
from utils import plot_learning_curve, plot_metrics, evaluate_policy, save_model

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)
    
    def __len__(self):
        return len(self.buffer)

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(QNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, state):
        return self.network(state)

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, epsilon_start=1.0, 
                 epsilon_end=0.05, epsilon_decay=0.995, buffer_size=10000, batch_size=64):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        
        # Epsilon-greedy exploration
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Networks and optimizer
        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Experience replay
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network.to(self.device)
        self.target_network.to(self.device)
    
    def select_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return 0  # Not enough samples
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute Q values
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = nn.MSELoss()(q_values, target_q_values)
        
        # Update network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def update_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def get_policy(self):
        """Return a policy function for evaluation"""
        def policy(state):
            return self.select_action(state, training=False)
        return policy

def train_dqn(env, num_episodes=1000, target_update_freq=10, print_freq=10, eval_freq=100):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = DQNAgent(state_dim, action_dim)
    
    # Training metrics
    episode_rewards = []
    losses = []
    all_moisture_levels = []
    all_actions = []
    
    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0
        episode_loss = 0
        done = False
        
        while not done:
            # Select action
            action = agent.select_action(state)
            
            # Take step
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store transition
            agent.replay_buffer.add(state, action, reward, next_state, done)
            
            # Update network
            loss = agent.update()
            if loss != 0:
                episode_loss += loss
            
            # Update state and reward
            state = next_state
            episode_reward += reward
        
        # Update target network
        if episode % target_update_freq == 0:
            agent.update_target_network()
        
        # Update exploration rate
        agent.update_epsilon()
        
        # Store episode metrics
        episode_rewards.append(episode_reward)
        losses.append(episode_loss)
        all_moisture_levels.append(env.moisture_history)
        all_actions.append(env.action_history)
        
        # Print progress
        if episode % print_freq == 0:
            print(f"Episode {episode}, Reward: {episode_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
        
        # Evaluate policy
        if episode % eval_freq == 0 or episode == num_episodes:
            policy = agent.get_policy()
            avg_reward, success_rate, avg_water_usage = evaluate_policy(env, policy)
            print(f"Evaluation - Avg Reward: {avg_reward:.2f}, Success Rate: {success_rate:.2f}%, Water Usage: {avg_water_usage:.2f}")
    
    # Save trained model
    save_model(agent.q_network, "DQN")
    
    # Plot learning curve
    plot_learning_curve(episode_rewards, "DQN")
    
    # Plot final metrics
    plot_metrics("DQN", episode_rewards, all_moisture_levels, all_actions)
    
    return agent, episode_rewards, all_moisture_levels, all_actions

if __name__ == "__main__":
    # Create the environment
    env = SmartIrrigationEnv()
    
    # Create directories for results
    os.makedirs("results/DQN", exist_ok=True)
    
    # Train the agent
    agent, rewards, moisture_levels, actions = train_dqn(env, num_episodes=500)
    
    # Final evaluation
    policy = agent.get_policy()
    avg_reward, success_rate, avg_water_usage = evaluate_policy(env, policy, episodes=20)
    
    print("\nFinal Evaluation Results:")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Success Rate (moisture in optimal range): {success_rate:.2f}%")
    print(f"Average Water Usage: {avg_water_usage:.2f}")
    
    # Plot final episode
    state, _ = env.reset()
    done = False
    
    while not done:
        action = agent.select_action(state, training=False)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    
    fig = env.plot_episode("Final")
    plt.savefig("results/DQN/final_episode.png")
    plt.close() 