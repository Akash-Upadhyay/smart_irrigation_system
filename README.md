# Smart Irrigation System using Reinforcement Learning

## Problem Statement

The goal of this project is to develop a smart irrigation system that optimizes water usage while maintaining optimal soil moisture levels. The system should adapt to changing environmental conditions and make decisions on the amount of water to apply based on current and forecasted weather conditions. This is achieved using reinforcement learning (RL) algorithms.

## MDP Formulation

The problem is formulated as a Markov Decision Process (MDP) with the following components:

### State Space
The state space consists of the following variables:
- **Soil moisture level** (0-100%): Current moisture level of the soil.
- **Temperature** (°C): Current temperature.
- **Humidity** (%): Current humidity level.
- **Precipitation forecast** (0-100%): Forecasted precipitation.
- **Time of day** (0-23 hours): Current hour of the day.
- **Day of week** (0-6): Current day of the week.

### Action Space
The agent can take 4 discrete actions representing different irrigation levels:
- **0**: No irrigation
- **1**: Low irrigation (5 units)
- **2**: Medium irrigation (15 units)
- **3**: High irrigation (25 units)

### Reward Function
The reward function is designed to encourage optimal soil moisture and water conservation:
- **Moisture reward**: +10 if moisture is in the optimal range (60-80%), otherwise negative based on deviation.
- **Water usage penalty**: -0.5 * water_amount to encourage water conservation.
- **Overwatering penalty**: -2 * water_amount if precipitation > 20% to avoid wasteful irrigation.

### Transition Dynamics
- **Soil moisture**: Increases with irrigation and precipitation, decreases with evaporation (dependent on temperature and humidity).
- **Weather conditions**: Evolve stochastically based on realistic patterns.
- **Time and day**: Advance deterministically.

## Directory Structure

- `environment.py`: Custom Gym environment for the smart irrigation system.
- `mdp_formulation.py`: Detailed description of the MDP formulation.
- `dqn.py`: Implementation of the Deep Q-Learning algorithm.
- `actor_critic.py`: Implementation of the Actor-Critic algorithm.
- `ppo.py`: Implementation of the Proximal Policy Optimization algorithm.
- `utils.py`: Utility functions for plotting and evaluation.
- `compare_algorithms.py`: Script to train and compare all three algorithms.
- `requirements.txt`: List of dependencies required to run the project.

## Approaches Used

### 1. Deep Q-Learning (DQN)
A value-based method that learns the optimal action-value function (Q-function). Key features include:
- **Experience replay**: Breaks correlations between consecutive samples.
- **Target network**: Stabilizes learning.
- **Epsilon-greedy exploration**: Balances exploration and exploitation.

### 2. Actor-Critic
A hybrid approach combining policy-based and value-based methods:
- **Actor**: Updates the policy directly.
- **Critic**: Evaluates the policy by estimating the value function.
- **Advantage-based policy updates**: Improves learning efficiency.

### 3. Proximal Policy Optimization (PPO)
A policy optimization method using a clipped surrogate objective for stable updates:
- **Trust region policy optimization**: Ensures stable updates without high computational cost.
- **Generalized Advantage Estimation (GAE)**: Improves sample efficiency.
- **Multiple epochs**: Optimizes policy for each batch of data.

## Results and Comparison

The algorithms are evaluated based on:
1. **Average reward**: Measures overall performance.
2. **Success rate**: Percentage of time the soil moisture is within the optimal range.
3. **Water usage efficiency**: Measures water conservation.

### Comparison
- **DQN**: Often shows steady improvement but may plateau after sufficient training.
- **Actor-Critic**: Sometimes has higher variance in learning but can reach good policies.
- **PPO**: Often more sample-efficient, potentially reaching higher rewards with fewer episodes.

### Reasoning for Results
- **DQN**: Benefits from experience replay and target networks, but may struggle with continuous state spaces.
- **Actor-Critic**: Balances exploration and exploitation well, but can be sensitive to hyperparameters.
- **PPO**: Provides stable updates and is robust to hyperparameter settings, often leading to better performance.

## Running the Code

### Prerequisites
- Python 3.8+
- PyTorch
- Gymnasium
- NumPy
- Matplotlib

### Installation
```bash
pip install -r requirements.txt
```

### Execution Steps

To train and compare all three algorithms:
```bash
python compare_algorithms.py
```

To train a specific algorithm:
```bash
python dqn.py  # For DQN
python actor_critic.py  # For Actor-Critic
python ppo.py  # For PPO
```

## Results and Plots
Results and plots are saved in the `results` directory:
- **Individual algorithm results**: `results/[algorithm_name]/`
- **Comparison results**: `results/comparison/`

These include learning curves, performance metrics, and final irrigation strategies. 