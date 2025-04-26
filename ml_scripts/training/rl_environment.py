# ml_scripts/rl_environment.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class FinancialPlannerEnv(gym.Env):
    metadata = {"render_modes": [], "render_fps": 4}

    def __init__(self, user_profile, goal_amount, time_steps, investment_options):
        super().__init__()
        self.start_profile = user_profile # Dict with initial Age, Income, Savings etc.
        self.goal_amount = goal_amount
        self.total_time_steps = time_steps # e.g., number of months
        self.investment_options = investment_options # Dict with 'avg_return', 'volatility' per type

        # --- Define Action Space (Example: Save %, Allocate % to 2 options) ---
        # Action: [save_pct, alloc_inv1_pct] (alloc_inv2 = 1 - alloc_inv1)
        # Bounds: Save 0-50%, Alloc 0-100%
        self.action_space = spaces.Box(low=np.array([0, 0]), high=np.array([0.5, 1.0]), dtype=np.float32)

        # --- Define Observation Space (Example: simplified state) ---
        # State: [current_savings, current_inv_value, time_steps_left]
        # Define reasonable bounds (adjust based on expected values)
        low_bounds = np.array([0, 0, 0])
        high_bounds = np.array([goal_amount * 5, goal_amount * 5, time_steps + 1], dtype=np.float32) # Example high bounds
        self.observation_space = spaces.Box(low=low_bounds, high=high_bounds, dtype=np.float32)

        # Initial state variables (will be reset)
        self.current_savings = 0
        self.current_investment_value = 0
        self.current_step = 0

        print("FinancialPlannerEnv initialized.")

    def _get_obs(self):
        """Returns the current state observation."""
        return np.array([self.current_savings, self.current_investment_value,
                         self.total_time_steps - self.current_step], dtype=np.float32)

    def _get_info(self):
        """Returns auxiliary information (optional)."""
        return {"current_step": self.current_step}

    def reset(self, seed=None, options=None):
        """Resets the environment to the initial state."""
        super().reset(seed=seed)
        print("Resetting environment...")
        # Initialize state based on starting profile (simplified example)
        self.current_savings = self.start_profile.get("InitialSavings", 10000) # Get initial savings or default
        self.current_investment_value = self.start_profile.get("InitialInvestments", 5000) # Example
        self.current_step = 0

        observation = self._get_obs()
        info = self._get_info()
        print(f"Reset complete. Initial Obs: {observation}")
        return observation, info

    def step(self, action):
        """Applies action, simulates time step, calculates reward."""
        self.current_step += 1
        print(f"\n--- Step {self.current_step} ---")
        print(f"Action taken: {action}")

        save_pct = action[0]
        alloc_inv1_pct = action[1]
        alloc_inv2_pct = 1.0 - alloc_inv1_pct # Assuming 2 investment choices for simplicity

        # --- Simulate Savings ---
        # Simplified: Add saving based on a fraction of 'income' (needs income in profile)
        monthly_income = self.start_profile.get("MonthlyIncomeEstimate", 50000) # Example income
        saved_this_step = monthly_income * save_pct
        self.current_savings += saved_this_step
        print(f"Savings: Added {saved_this_step:.2f}, Total Savings: {self.current_savings:.2f}")

        # --- Simulate Investment ---
        # Simplified: Assume savings are invested immediately based on allocation
        # Calculate growth/loss based on stochastic returns (mean + random noise * volatility)
        # This needs a more robust simulation model!
        # Example placeholder: just add a small fixed % for now
        inv_growth_factor = 0.005 # Placeholder monthly growth
        invested_amount = self.current_investment_value + saved_this_step # Invest new savings too
        # More complex: calculate weighted return based on alloc_inv1/2_pct and their simulated returns
        self.current_investment_value = invested_amount * (1 + inv_growth_factor + np.random.normal(0, 0.01)) # Add noise
        self.current_savings = 0 # Assume all savings are invested for simplicity here
        print(f"Investment Value: {self.current_investment_value:.2f}")


        # --- Calculate Reward ---
        reward = 0
        current_total_value = self.current_savings + self.current_investment_value

        # Reward progress towards goal
        progress = current_total_value / self.goal_amount
        reward += progress * 0.1 # Small reward for progress

        # --- Check Termination/Truncation ---
        terminated = current_total_value >= self.goal_amount
        truncated = self.current_step >= self.total_time_steps

        if terminated:
            reward += 100 # Large reward for reaching goal
            print("GOAL REACHED!")
        elif truncated:
            reward -= 50 # Penalty for running out of time
            print("TIME LIMIT REACHED.")
            # Penalty based on how far off the goal?
            reward -= (self.goal_amount - current_total_value) / self.goal_amount * 10

        # Optional: Small penalty for negative returns? Small reward for saving action?
        # reward -= max(0, previous_inv_value - self.current_investment_value) * 0.01 # Penalty for loss
        reward += save_pct * 0.5 # Small reward for saving effort


        observation = self._get_obs()
        info = self._get_info()

        print(f"Reward this step: {reward:.3f}")
        return observation, reward, terminated, truncated, info

    def close(self):
        print("Closing environment.")
        pass

# --- Example Usage (for testing the environment) ---
if __name__ == '__main__':
    # Dummy data for testing
    profile = {"InitialSavings": 5000, "InitialInvestments": 10000, "MonthlyIncomeEstimate": 60000}
    goal = 1000000
    steps = 120 # 10 years
    # Dummy investment options (replace with actual simulated characteristics)
    options = {"Inv1": {"avg_return": 0.08, "volatility": 0.15}, "Inv2": {"avg_return": 0.04, "volatility": 0.05}}

    env = FinancialPlannerEnv(profile, goal, steps, options)
    # Test reset
    obs, info = env.reset()
    print("Initial Observation:", obs)
    print("Initial Info:", info)

    # Test step with random action
    random_action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(random_action)
    print("Observation after step:", obs)
    print("Reward:", reward)
    print("Terminated:", terminated)
    print("Truncated:", truncated)
    print("Info:", info)

    env.close()