import numpy as np
import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers import GymWrapper
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
import wandb
from wandb.integration.sb3 import WandbCallback

def make_env(horizon: int = 1000, render: bool = False):

    controller_config = load_composite_controller_config(controller="BASIC")

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


def train():

    run = wandb.init(
        project="robosuite",
        sync_tensorboard=True
    )

    env = make_vec_env(make_env, env_kwargs={"horizon": 1000, "render": False}, n_envs=8, seed=42)

    model = SAC("MlpPolicy", env, tensorboard_log=f"runs/{run.id}")
    callback = WandbCallback(
        model_save_path="models/robosuite_lift",
        model_save_freq=50_000,
        gradient_save_freq=100,
        verbose=1
    )
    model.learn(total_timesteps=1_000_000, callback=callback)
    model.save("models/robosuite_lift/final.zip")


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
    eval()


if __name__ == "__main__":
    main()
