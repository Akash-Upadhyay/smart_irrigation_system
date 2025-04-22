import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import os

from environment import SmartIrrigationEnv
from utils import plot_learning_curve, plot_metrics, evaluate_policy, save_model

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class ActorCriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ActorCriticNetwork, self).__init__()
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state):
        features = self.shared(state)
        action_probs = self.actor(features)
        value = self.critic(features)
        return action_probs, value

class ActorCriticAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99):
        self.gamma = gamma
        self.action_dim = action_dim
        
        # Actor-Critic network and optimizer
        self.network = ActorCriticNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network.to(self.device)
        
        # Storage for trajectory
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.action_probs = []
    
    def select_action(self, state, training=True):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_probs, value = self.network(state_tensor)
            
            if training:
                # Sample from action distribution during training
                distribution = Categorical(action_probs)
                action = distribution.sample().item()
                
                # Store trajectory information
                self.action_probs.append(action_probs[0, action].item())
                self.values.append(value.item())
            else:
                # Take greedy action during evaluation
                action = torch.argmax(action_probs).item()
            
            return action
    
    def store_transition(self, state, action, reward, done):
        """Store trajectory information for updating the policy"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
    
    def update(self):
        # Convert lists to tensors
        states = torch.FloatTensor(self.states).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        rewards = torch.FloatTensor(self.rewards).to(self.device)
        dones = torch.FloatTensor(self.dones).to(self.device)
        
        # Calculate returns and advantages
        returns = []
        advantages = []
        
        # Get value predictions for states
        action_probs, values = self.network(states)
        values = values.squeeze()
        
        # Calculate returns (discounted sum of future rewards)
        R = 0
        for i in reversed(range(len(rewards))):
            R = rewards[i] + self.gamma * R * (1 - dones[i])
            returns.insert(0, R)
        
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Calculate advantages (return - value)
        advantages = returns - values
        
        # Calculate losses
        # Actor loss (policy gradient)
        log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1))).squeeze(1)
        actor_loss = -(log_probs * advantages.detach()).mean()
        
        # Critic loss (value prediction)
        critic_loss = F.mse_loss(values, returns)
        
        # Entropy loss (for exploration)
        entropy = -(action_probs * torch.log(action_probs + 1e-10)).sum(dim=1).mean()
        entropy_coef = 0.01
        
        # Total loss
        total_loss = actor_loss + 0.5 * critic_loss - entropy_coef * entropy
        
        # Update network
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
        self.optimizer.step()
        
        # Clear trajectory buffers
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.action_probs = []
        
        return total_loss.item(), actor_loss.item(), critic_loss.item()
    
    def get_policy(self):
        """Return a policy function for evaluation"""
        def policy(state):
            return self.select_action(state, training=False)
        return policy

def train_actor_critic(env, num_episodes=1000, update_freq=1, print_freq=10, eval_freq=100):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = ActorCriticAgent(state_dim, action_dim)
    
    # Training metrics
    episode_rewards = []
    actor_losses = []
    critic_losses = []
    all_moisture_levels = []
    all_actions = []
    
    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        # Episode loop
        while not done:
            # Select action
            action = agent.select_action(state)
            
            # Take step
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store transition
            agent.store_transition(state, action, reward, done)
            
            # Update state and reward
            state = next_state
            episode_reward += reward
            
            # Update policy every update_frequency steps
            if done or (len(agent.states) % update_freq == 0):
                total_loss, actor_loss, critic_loss = agent.update()
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)
        
        # Store episode metrics
        episode_rewards.append(episode_reward)
        all_moisture_levels.append(env.moisture_history)
        all_actions.append(env.action_history)
        
        # Print progress
        if episode % print_freq == 0:
            print(f"Episode {episode}, Reward: {episode_reward:.2f}")
        
        # Evaluate policy
        if episode % eval_freq == 0 or episode == num_episodes:
            policy = agent.get_policy()
            avg_reward, success_rate, avg_water_usage = evaluate_policy(env, policy)
            print(f"Evaluation - Avg Reward: {avg_reward:.2f}, Success Rate: {success_rate:.2f}%, Water Usage: {avg_water_usage:.2f}")
    
    # Save trained model
    save_model(agent.network, "ActorCritic")
    
    # Plot learning curve
    plot_learning_curve(episode_rewards, "ActorCritic")
    
    # Plot final metrics
    plot_metrics("ActorCritic", episode_rewards, all_moisture_levels, all_actions)
    
    return agent, episode_rewards, all_moisture_levels, all_actions

if __name__ == "__main__":
    # Create the environment
    env = SmartIrrigationEnv()
    
    # Create directories for results
    os.makedirs("results/ActorCritic", exist_ok=True)
    
    # Train the agent
    agent, rewards, moisture_levels, actions = train_actor_critic(env, num_episodes=500)
    
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
    plt.savefig("results/ActorCritic/final_episode.png")
    plt.close() 