import gymnasium as gym
from gymnasium.wrappers import RecordVideo, FilterObservation, FlattenObservation
from minigrid.wrappers import ObservationWrapper, RGBImgObsWrapper
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure


class AgentTrainer:
    """
    Class to train RL agents using stable baselines ppo. Keeps track of checkpointing too.
    """
    def __init__(
            self,
            seed: int,
            device: str = "cpu",
            log_path: str = None,
            checkpoint_path: str = None,
            checkpoint_freq: int = 50_000,
    ):
        self.seed = seed
        self.device = device

        # Set up logging
        self.logger = None
        if log_path is not None:
            logger = configure(log_path, ["stdout", "csv"])
            self.logger = logger

        # Set up checkpointing
        self.checkpoint_callback = None
        if checkpoint_path is not None:
            self.checkpoint_callback = CheckpointCallback(
                save_freq=checkpoint_freq,
                save_path=checkpoint_path,
                name_prefix="checkpoint",
                save_replay_buffer=True
            )

    def train_agent(
        self,
        env: gym.Env,
        steps: int,
        verbose: int = 0
    ) -> PPO:
        """
        Trains a PPO agent on a provided gym environment.
        """
        model = PPO("MultiInputPolicy", env, verbose=verbose, seed=self.seed, device=self.device)

        # Logging
        if self.logger is not None:
            model.set_logger(self.logger)

        if self.checkpoint_callback is not None:
            model.learn(total_timesteps=steps, callback=self.checkpoint_callback)
        # Otherwise, just train normally
        else:
            model.learn(total_timesteps=steps)

        return model

    def evaluate_agent(self, agent: PPO, eval_env: gym.Env, n_episodes: int = 10, record_name: str = None) -> float:
        """
        Evaluates a trained agent on the provided env for a number of episodes.
        Sets the seed according to self.seed so that we get deterministic evaluation behavior.
        """

        if record_name is not None:
            eval_env = RecordVideo(
                eval_env,
                video_folder=f"./videos/{record_name}",
                episode_trigger=lambda x: True,
                name_prefix=record_name
            )

        total_reward = 0.0
        for episode in range(n_episodes):
            obs, info = eval_env.reset(seed=self.seed + episode)
            done = False
            while not done:
                action, _ = agent.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                done = terminated or truncated
                total_reward += reward

        eval_env.close()
        return total_reward / n_episodes


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
