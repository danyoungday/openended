import gymnasium as gym
from gymnasium.wrappers import RecordVideo, FilterObservation, FlattenObservation
from minigrid.wrappers import ObservationWrapper, RGBImgObsWrapper
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback


class AgentTrainer:
    """
    Class to train RL agents using stable baselines ppo. Keeps track of checkpointing too.
    """
    def __init__(self, seed: int, device: str = "cpu", checkpoint_name: str = None, checkpoint_freq: int = 100_000):
        self.seed = seed
        self.device = device

        self.checkpoint_callback = None
        if checkpoint_name is not None:
            self.checkpoint_callback = CheckpointCallback(
                save_freq=checkpoint_freq,
                save_path="./models",
                name_prefix=checkpoint_name
            )

    def train_agent(self, env: gym.Env, steps: int, verbose: int = 0) -> PPO:
        """
        Trains a PPO agent on a provided gym environment.
        """
        model = PPO("MultiInputPolicy", env, verbose=verbose, seed=self.seed, device=self.device)

        if self.checkpoint_callback is not None:
            model.learn(total_timesteps=steps, callback=self.checkpoint_callback)
        else:
            model.learn(total_timesteps=steps)

        return model


class MiniWrapper(gym.Wrapper):
    """
    Wrapper needed to make MiniGrid compatible with stable baselines. Converts observation to pixels and applies
    a custom wrapper if provided.
    """
    def __init__(self, env, custom_wrapper: gym.Wrapper = None):
        if custom_wrapper is not None:
            env = custom_wrapper(env)
        env = FilterObservation(env, filter_keys=["image"])
        env = RGBImgObsWrapper(env)
        super().__init__(env)

        # img_space = env.observation_space["image"]
        # direction_space = env.observation_space["direction"]

        # self._img_dim = img_space.shape[0] * img_space.shape[1] * img_space.shape[2]
        # self._direction_dim = direction_space.n

        # self.observation_space = gym.spaces.Box(
        #     low=0.0,
        #     high=1.0,
        #     shape=(self._img_dim + self._direction_dim,),
        #     dtype=np.float32
        # )

    # def observation(self, observation):
    #     """
    #     Override the observation function and flatten the image and normalize it then one hot the direction.
    #     """
    #     img = observation["image"].flatten() / 255.0
    #     direction = np.zeros(self._direction_dim, dtype=np.float32)
    #     direction[observation["direction"]] = 1.0
    #     return np.concatenate([img, direction])


def visualize_agent(agent: PPO, env: gym.Env, save_prefix: str):
    """
    Takes a trained agent and environment and runs an episode to save a video.
    """
    video_env = RecordVideo(env, video_folder="./videos/", name_prefix=save_prefix)

    obs, _ = video_env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = video_env.step(action)
        done = terminated or truncated
        total_reward += reward
    print(f"Total reward: {total_reward}")
    video_env.close()
