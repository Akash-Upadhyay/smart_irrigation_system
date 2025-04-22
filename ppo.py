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

class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size
    
    def store(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)
    
    def clear(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
    
    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        
        return np.array(self.states), \
               np.array(self.actions), \
               np.array(self.probs), \
               np.array(self.vals), \
               np.array(self.rewards), \
               np.array(self.dones), \
               batches

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ActorNetwork, self).__init__()
        
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, state):
        return self.actor(state)

class CriticNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=128):
        super(CriticNetwork, self).__init__()
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        return self.critic(state)

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, gae_lambda=0.95,
                 policy_clip=0.2, batch_size=64, n_epochs=10, entropy_coef=0.01):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        
        # Actor and Critic networks
        self.actor = ActorNetwork(state_dim, action_dim)
        self.critic = CriticNetwork(state_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # Memory
        self.memory = PPOMemory(batch_size)
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor.to(self.device)
        self.critic.to(self.device)
    
    def select_action(self, state, training=True):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_probs = self.actor(state_tensor)
            value = self.critic(state_tensor)
            
            if training:
                # Sample from action distribution during training
                distribution = Categorical(action_probs)
                action = distribution.sample().item()
                action_prob = action_probs[0, action].item()
                
                return action, action_prob, value.item()
            else:
                # Take greedy action during evaluation
                action = torch.argmax(action_probs).item()
                return action
    
    def learn(self):
        # Retrieve batch from memory
        states, actions, old_probs, values, rewards, dones, batches = self.memory.generate_batches()
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        old_probs = torch.FloatTensor(old_probs).to(self.device)
        values = torch.FloatTensor(values).to(self.device)
        
        # Calculate advantages using Generalized Advantage Estimation (GAE)
        advantages = np.zeros(len(rewards), dtype=np.float32)
        for t in range(len(rewards)-1):
            discount = 1
            a_t = 0
            for k in range(t, len(rewards)-1):
                a_t += discount * (rewards[k] + self.gamma * values[k+1] * (1-dones[k]) - values[k])
                discount *= self.gamma * self.gae_lambda
            advantages[t] = a_t
        
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        # Compute returns
        returns = advantages + values
        
        # Training loop
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        
        for _ in range(self.n_epochs):
            for batch in batches:
                # Get minibatch
                batch_states = states[batch]
                batch_actions = actions[batch]
                batch_old_probs = old_probs[batch]
                batch_advantages = advantages[batch]
                batch_returns = returns[batch]
                
                # Get current action probabilities and values
                action_probs = self.actor(batch_states)
                values = self.critic(batch_states).squeeze()
                
                # Calculate ratio for PPO
                distribution = Categorical(action_probs)
                entropy = distribution.entropy().mean()
                new_probs = distribution.log_prob(batch_actions)
                old_probs_log = torch.log(batch_old_probs + 1e-10)
                
                # Calculate probability ratio
                ratio = torch.exp(new_probs - old_probs_log)
                
                # Calculate surrogate losses
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.policy_clip, 1.0 + self.policy_clip) * batch_advantages
                
                # Calculate actor loss
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Calculate critic loss
                critic_loss = F.mse_loss(values, batch_returns)
                
                # Calculate total loss (with entropy for exploration)
                total_loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy
                
                # Update actor
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.item()
        
        # Clear memory after update
        self.memory.clear()
        
        return total_actor_loss / (self.n_epochs * len(batches)), \
               total_critic_loss / (self.n_epochs * len(batches)), \
               total_entropy / (self.n_epochs * len(batches))
    
    def get_policy(self):
        """Return a policy function for evaluation"""
        def policy(state):
            return self.select_action(state, training=False)
        return policy

def train_ppo(env, num_episodes=1000, update_timestep=2048, print_freq=10, eval_freq=100):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = PPOAgent(state_dim, action_dim)
    
    # Training metrics
    episode_rewards = []
    actor_losses = []
    critic_losses = []
    entropies = []
    all_moisture_levels = []
    all_actions = []
    
    # Training variables
    time_step = 0
    
    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        # Episode loop
        while not done:
            # Select action
            action, action_prob, value = agent.select_action(state)
            
            # Take step
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store in memory
            agent.memory.store(state, action, action_prob, value, reward, done)
            
            # Update time step and state
            time_step += 1
            state = next_state
            episode_reward += reward
            
            # Update policy if enough steps
            if time_step % update_timestep == 0:
                actor_loss, critic_loss, entropy = agent.learn()
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)
                entropies.append(entropy)
        
        # Learn at the end of episode if not updated
        if len(agent.memory.states) > 0:
            actor_loss, critic_loss, entropy = agent.learn()
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)
            entropies.append(entropy)
        
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
    save_model(agent.actor, "PPO")
    
    # Plot learning curve
    plot_learning_curve(episode_rewards, "PPO")
    
    # Plot final metrics
    plot_metrics("PPO", episode_rewards, all_moisture_levels, all_actions)
    
    return agent, episode_rewards, all_moisture_levels, all_actions

if __name__ == "__main__":
    # Create the environment
    env = SmartIrrigationEnv()
    
    # Create directories for results
    os.makedirs("results/PPO", exist_ok=True)
    
    # Train the agent
    agent, rewards, moisture_levels, actions = train_ppo(env, num_episodes=500)
    
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
    plt.savefig("results/PPO/final_episode.png")
    plt.close() 