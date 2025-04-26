## ml_scripts/training/train_rl_model.py
import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
# Import your custom environment
try:
    # Direct import since it's in the same directory
    from ml_scripts.training.rl_environment import FinancialPlannerEnv # <--- CORRECTED DIRECT IMPORT
except ImportError as e:
    print(f"Error importing FinancialPlannerEnv: {e}")
    print("Make sure rl_environment.py exists in the ml_scripts/training/ directory.")
    exit()

# --- Configuration ---
# ...(rest of the script remains the same)...
# --- Configuration ---
# Dummy data for training initialization (should match env needs)
profile = {"InitialSavings": 5000, "InitialInvestments": 10000, "MonthlyIncomeEstimate": 60000}
goal = 1000000
steps = 240 # Increase steps for more learning (e.g., 20 years)
options = {"Inv1": {"avg_return": 0.08, "volatility": 0.15}, "Inv2": {"avg_return": 0.04, "volatility": 0.05}}

# Where to save the trained model
MODELS_DIR = '../../models'
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, 'rl_planner_ppo_v1') # Version your models
LOG_DIR = './rl_logs/' # For TensorBoard logs (optional but good)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

TOTAL_TIMESTEPS = 100000 # Start small (e.g., 100k), increase significantly for real results (e.g., 1M+)

# --- Create Environment ---
# You can wrap the environment creation if needed
def make_env():
    return FinancialPlannerEnv(profile, goal, steps, options)

# Vectorized environments often speed up training
num_cpu = 4 # Use multiple cores if available
vec_env = make_vec_env(make_env, n_envs=num_cpu)

# --- Define and Train Agent ---
# PPO is a good starting point for continuous/box action spaces
# Adjust hyperparameters (learning_rate, n_steps, batch_size, gamma etc.) based on results
model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log=LOG_DIR,
            learning_rate=0.0003, n_steps=2048, batch_size=64, gamma=0.99)

print(f"Starting training for {TOTAL_TIMESTEPS} timesteps...")
# This is the main training loop. It will take time.
model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True)

# --- Save the Trained Agent ---
print(f"Training complete. Saving model to {MODEL_SAVE_PATH}...")
model.save(MODEL_SAVE_PATH)
print("Model saved.")

# Close the environment
vec_env.close()

print("\n--- RL Agent Training Finished ---")
print(f"To monitor training (optional), run: tensorboard --logdir {LOG_DIR}")
