import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from gymnasium import spaces
import random

class SmartIrrigationEnv(gym.Env):
    """
    A smart irrigation environment for reinforcement learning.
    
    State space:
    - Soil moisture level (0-100%)
    - Temperature (°C)
    - Humidity (%)
    - Precipitation forecast (0-100%)
    - Time of day (0-23 hours)
    - Day of week (0-6)
    
    Action space:
    - Water amount (0: None, 1: Low, 2: Medium, 3: High)
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}
    
    def __init__(self, render_mode=None):
        super(SmartIrrigationEnv, self).__init__()
        
        # Define action and observation space
        # Actions: Water amount (discrete)
        self.action_space = spaces.Discrete(4)  # 0: None, 1: Low, 2: Medium, 3: High
        
        # State space
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0]),  # soil moisture, temp, humidity, precipitation, time, day
            high=np.array([100, 50, 100, 100, 23, 6]),
            dtype=np.float32
        )
        
        # Environment parameters
        self.max_steps = 30  # 30 days simulation
        self.step_count = 0
        self.render_mode = render_mode
        
        # Plant parameters
        self.optimal_moisture_lower = 60
        self.optimal_moisture_upper = 80
        
        # History for plotting
        self.moisture_history = []
        self.action_history = []
        self.reward_history = []
        
        # Initialize state
        self.state = self._get_initial_state()
        
    def _get_initial_state(self):
        # Random initial state
        soil_moisture = np.random.uniform(30, 70)
        temperature = np.random.uniform(15, 35)
        humidity = np.random.uniform(30, 80)
        precipitation = np.random.uniform(0, 20)
        time_of_day = np.random.randint(6, 20)  # Daytime hours
        day_of_week = np.random.randint(0, 7)
        
        return np.array([soil_moisture, temperature, humidity, precipitation, time_of_day, day_of_week], dtype=np.float32)
    
    def step(self, action):
        # Unpack state
        soil_moisture, temperature, humidity, precipitation, time_of_day, day_of_week = self.state
        
        # Apply action (irrigation amount)
        if action == 0:  # No irrigation
            water_amount = 0
        elif action == 1:  # Low irrigation
            water_amount = 5
        elif action == 2:  # Medium irrigation
            water_amount = 15
        else:  # High irrigation
            water_amount = 25
        
        # Update soil moisture based on action, evaporation, and precipitation
        evaporation = (0.1 * temperature) + (0.05 * (100 - humidity))
        moisture_gain_from_rain = 0.5 * precipitation
        
        # Update moisture level
        new_soil_moisture = soil_moisture + water_amount + moisture_gain_from_rain - evaporation
        new_soil_moisture = np.clip(new_soil_moisture, 0, 100)
        
        # Simulate next day conditions
        new_temperature = np.clip(temperature + np.random.uniform(-5, 5), 0, 50)
        new_humidity = np.clip(humidity + np.random.uniform(-10, 10), 0, 100)
        new_precipitation = np.clip(np.random.exponential(5) if np.random.random() < 0.3 else 0, 0, 100)
        new_time_of_day = (time_of_day + 1) % 24
        new_day_of_week = (day_of_week + 1) % 7 if new_time_of_day == 0 else day_of_week
        
        # Calculate reward
        # Reward for keeping moisture in optimal range
        if self.optimal_moisture_lower <= new_soil_moisture <= self.optimal_moisture_upper:
            moisture_reward = 10
        else:
            moisture_reward = -abs(new_soil_moisture - (self.optimal_moisture_lower + self.optimal_moisture_upper) / 2)
        
        # Penalty for water usage (water conservation)
        water_penalty = -0.5 * water_amount
        
        # Penalty for overwatering when precipitation is high
        overwatering_penalty = -2 * water_amount if precipitation > 20 else 0
        
        reward = moisture_reward + water_penalty + overwatering_penalty
        
        # Update state
        self.state = np.array([
            new_soil_moisture, new_temperature, new_humidity, 
            new_precipitation, new_time_of_day, new_day_of_week
        ], dtype=np.float32)
        
        # Update step count and check if episode is done
        self.step_count += 1
        terminated = self.step_count >= self.max_steps
        truncated = False
        
        # Store history for plotting
        self.moisture_history.append(new_soil_moisture)
        self.action_history.append(action)
        self.reward_history.append(reward)
        
        # Return step information
        return self.state, reward, terminated, truncated, {}
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset environment state
        self.state = self._get_initial_state()
        self.step_count = 0
        
        # Reset history
        self.moisture_history = []
        self.action_history = []
        self.reward_history = []
        
        return self.state, {}
    
    def render(self):
        if self.render_mode == "human":
            # Implement visualization if needed
            pass
        return None
    
    def close(self):
        pass
    
    def plot_episode(self, episode):
        """Plot the moisture levels, actions, and rewards for the episode"""
        fig, axs = plt.subplots(3, 1, figsize=(10, 12))
        
        # Plot moisture levels
        axs[0].plot(self.moisture_history)
        axs[0].axhline(y=self.optimal_moisture_lower, color='g', linestyle='--', label='Optimal Lower')
        axs[0].axhline(y=self.optimal_moisture_upper, color='g', linestyle='--', label='Optimal Upper')
        axs[0].set_ylabel('Soil Moisture (%)')
        axs[0].set_title(f'Episode {episode} - Soil Moisture Levels')
        axs[0].legend()
        
        # Plot actions
        axs[1].step(range(len(self.action_history)), self.action_history)
        axs[1].set_ylabel('Irrigation Action')
        axs[1].set_title('Irrigation Actions Taken')
        axs[1].set_yticks([0, 1, 2, 3])
        axs[1].set_yticklabels(['None', 'Low', 'Medium', 'High'])
        
        # Plot rewards
        axs[2].plot(self.reward_history)
        axs[2].set_xlabel('Step')
        axs[2].set_ylabel('Reward')
        axs[2].set_title('Rewards')
        
        plt.tight_layout()
        return fig 