import gymnasium as gym
from gymnasium.wrappers import RecordVideo

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


class AgentTrainer:

    def __init__(self, seed: int, device: str = "cpu"):
        self.seed = seed
        self.device = device

    def train_agent(self, env: gym.Env, steps: int, verbose: int = 0) -> PPO:
        model = PPO("MlpPolicy", env, verbose=verbose, seed=self.seed, device=self.device)
        model.learn(total_timesteps=steps)
        return model


def main():

    trainer = AgentTrainer(seed=42, device="cpu")

    # Train in parallel with a vecenv
    envs = make_vec_env("BipedalWalker-v3", n_envs=8, seed=42)
    agent = trainer.train_agent(envs, steps=500_000, verbose=1)

    # Save agent
    agent.save("bipedalwalker_agent")

    # Evaluate and record video
    env_view = RecordVideo(gym.make("BipedalWalker-v3", render_mode="rgb_array"), video_folder="./videos/", name_prefix="test")
    obs, _ = env_view.reset()
    done = False
    total_reward = 0
    while not done:
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env_view.step(action)
        done = terminated or truncated
        total_reward += reward
    
    print(f"Total reward: {total_reward}")
    env_view.close()

if __name__ == "__main__":
    main()
