"""
Smart Irrigation System - MDP Formulation

This file contains the description of the Markov Decision Process (MDP) for the
smart irrigation system problem.
"""

# MDP Components
# --------------

# 1. States (S):
# The state space consists of the following variables:
# - Soil moisture level (0-100%)
# - Temperature (°C)
# - Humidity (%)
# - Precipitation forecast (0-100%)
# - Time of day (0-23 hours)
# - Day of week (0-6)
#
# These variables represent the complete observable state of the environment.
# The state space is continuous for the first four variables and discrete for time and day.

# 2. Actions (A):
# The action space consists of the amount of irrigation water to apply:
# - 0: No irrigation
# - 1: Low irrigation (5 units)
# - 2: Medium irrigation (15 units)
# - 3: High irrigation (25 units)
#
# This represents a discrete action space with 4 possible actions.

# 3. Transition Function (P):
# The transition function P(s'|s,a) gives the probability of transitioning from state s 
# to state s' after taking action a.
#
# In our environment, the transitions follow these dynamics:
# - Soil moisture increases based on irrigation amount and precipitation
# - Soil moisture decreases due to evaporation, which depends on temperature and humidity
# - Temperature, humidity, and precipitation evolve stochastically based on realistic patterns
# - Time and day advance deterministically
#
# The transition function is described in detail in the environment.py file.

# 4. Reward Function (R):
# The reward function R(s,a,s') is designed to encourage:
# a. Maintaining soil moisture in the optimal range (60-80%)
# b. Water conservation (penalty for water usage)
# c. Avoiding overwatering when precipitation is forecasted
#
# The reward has three components:
# - Moisture reward: +10 if moisture is in optimal range, otherwise negative based on deviation
# - Water usage penalty: -0.5 * water_amount
# - Overwatering penalty: -2 * water_amount if precipitation > 20%
#
# The total reward is the sum of these components.

# 5. Discount Factor (γ):
# We use a discount factor close to 1 (γ = 0.99) since we care about long-term 
# soil health and water conservation.

# 6. Horizon:
# Episodes are finite with a horizon of 30 steps, simulating a 30-day period.

# RL Objective
# ------------
# The objective is to find a policy π(a|s) that maximizes the expected 
# discounted cumulative reward:
#
# J(π) = E_π [ Σ γ^t * R(s_t, a_t, s_{t+1}) ]
#
# This means finding a policy that:
# 1. Keeps soil moisture in the optimal range
# 2. Minimizes water usage
# 3. Adapts to weather conditions

# RL Challenges
# -------------
# 1. Exploration vs. Exploitation: The agent must explore different watering 
#    strategies while exploiting known effective strategies.
#
# 2. Delayed Rewards: The effects of irrigation on soil moisture persist over time.
#
# 3. Stochasticity: Weather conditions are stochastic, requiring robust policies.
#
# 4. Continuous State Space: The continuous state variables require function approximation.

# Implementation
# -------------
# We implement three different algorithms to solve this MDP:
# 1. Deep Q-Learning: Value-based method for discrete actions
# 2. Actor-Critic: Combined policy and value learning
# 3. Proximal Policy Optimization (PPO): Policy optimization with trust region constraint
#
# See the respective Python files for implementation details. 