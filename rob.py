# from gymnasium.wrappers import NormalizeObservation, NormalizeReward
# import numpy as np
import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers import GymWrapper
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env
import wandb
from wandb.integration.sb3 import WandbCallback

def make_env(horizon: int = 1000, render: bool = False):

    controller_config = load_composite_controller_config(controller="BASIC", robot="Panda")
    for key in ["left", "torso", "head", "base", "legs", "left", "torso", "head", "base", "legs"]:
        controller_config["body_parts"].pop(key, None)

    return GymWrapper(suite.make(
        env_name="Lift",
        robots="Panda",
        controller_configs=controller_config,
        has_renderer=render,
        reward_shaping=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        horizon=horizon,
        control_freq=20
    ))


def train(total_timesteps: int = 500_000):
    run = wandb.init(
        project="robosuite",
        sync_tensorboard=True
    )

    print("Creating env...")
    n_envs = 16
    env = make_vec_env(make_env, env_kwargs={"horizon": 1000, "render": False}, n_envs=n_envs, seed=42)
    env = VecNormalize(env, training=True, norm_obs=True, norm_reward=True)

    print("Creating model...")
    model = SAC("MlpPolicy", env, tensorboard_log=f"runs/{run.id}", verbose=1)
    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, 50_000 // n_envs),
        save_path='models/robosuite_lift',
        name_prefix='robosuite_lift_checkpoint',
        save_vecnormalize=True,
        verbose=2
    )
    wandb_callback = WandbCallback(
        verbose=1
    )

    print("Training...")
    model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback, wandb_callback])
    # model.save("models/robosuite_lift/final.zip")


def eval():

    model = SAC.load("models/robosuite_lift/model.zip")

    vis_env = make_env(horizon=1000, render=True)

    obs, info = vis_env.reset()
    done = False
    total_reward = 0
    while not done:
        action = model.predict(obs, deterministic=True)[0]
        obs, reward, done, truncated, info = vis_env.step(action)
        total_reward += reward
        vis_env.render()  # render the environment

    print("Total Reward:", total_reward)


def main():
    train()
    # eval()


if __name__ == "__main__":
    main()
