import gymnasium as gym
from gymnasium.wrappers import FilterObservation
import minigrid
from minigrid.wrappers import RGBImgObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from train_agent import AgentTrainer, visualize_agent
# from llm import LLMAgent, format_video_input


def check():
    env = gym.make("MiniGrid-DoorKey-5x5-v0", render_mode="human")
    env = RGBImgObsWrapper(env)
    env = FilterObservation(env, filter_keys=["image"])
    obs, info = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(obs)
        done = terminated or truncated
        break

class MiniWrapper(gym.Wrapper):
    def __init__(self, env):
        env = RGBImgObsWrapper(env)
        env = FilterObservation(env, filter_keys=["image"])
        super().__init__(env)

def main():
    env = make_vec_env("MiniGrid-DoorKey-5x5-v0", wrapper_class=MiniWrapper, n_envs=8, seed=42)
    trainer = AgentTrainer(seed=42, device="cpu")
    agent = trainer.train_agent(env, steps=100_000, verbose=1)

    # Save agent
    agent.save("models/minigrid_doorkey_agent")

    agent = PPO.load("models/minigrid_doorkey_agent")

    vis_env = gym.make("MiniGrid-DoorKey-5x5-v0", render_mode="rgb_array")
    vis_env = MiniWrapper(vis_env)
    visualize_agent(agent, vis_env, save_prefix="minigrid_doorkey_agent")


if __name__ == "__main__":
    # check()
    main()
