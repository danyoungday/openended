import gymnasium as gym
import panda_gym
from stable_baselines3 import HerReplayBuffer
from sb3_contrib import TQC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecVideoRecorder
from stable_baselines3.common.env_util import make_vec_env
import wandb
from wandb.integration.sb3 import WandbCallback
import yaml


def load_zoo_params(param_path: str):
    params = {}
    with open(param_path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    config = params["PandaReach-v1"]
    config["replay_buffer_class"] = HerReplayBuffer
    n_timesteps = config.pop("n_timesteps")
    policy = config.pop("policy")
    normalize = config.pop("normalize")
    return config, policy, n_timesteps, normalize


def train(save_path: str, total_timesteps: int, save_freq: int, n_envs: int):
    run = wandb.init(
        project="robosuite",
        sync_tensorboard=True
    )

    print("Creating env...")
    n_envs = 16
    env = make_vec_env("PandaReach-v3", n_envs=n_envs, seed=42)
    env = VecNormalize(env, training=True, norm_obs=True, norm_reward=True)

    print("Creating model...")
    config, policy, n_timesteps, normalize = load_zoo_params("hyperparams.yaml")
    print(config)
    model = TQC(policy, env, tensorboard_log=f"runs/{run.id}", verbose=0, **config)
    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, save_freq // n_envs),
        save_path=save_path,
        name_prefix='checkpoint',
        save_vecnormalize=True,
        verbose=2
    )
    wandb_callback = WandbCallback(
        verbose=0
    )

    model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback, wandb_callback])
    # model.save("models/robosuite_lift/final.zip")


def eval(video_save_path: str, n_repeats: int, model_save_path: str, vecnormalize_save_path: str):
    # config, policy, n_timesteps, normalize = load_zoo_params("hyperparams.yaml")

    vis_env = DummyVecEnv([lambda: gym.make("PandaReach-v3", render_mode="rgb_array")])
    vis_env = VecNormalize.load(vecnormalize_save_path, vis_env)

    vis_env.training = False
    vis_env.norm_reward = False
    vis_env = VecVideoRecorder(vis_env, "videos", record_video_trigger=lambda x: True, name_prefix=video_save_path)

    model = TQC.load(model_save_path, env=vis_env)

    # 5. Run the Evaluation Loop
    total_success = 0
    for episode in range(n_repeats):
        obs = vis_env.reset()
        done = False

        print("Recording video...")

        # We use a simple loop, but because we are using VecEnv, 'done' is technically
        # an array of booleans. For a single env, we check done[0].
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = vis_env.step(action)

            if done[0] or info[0].get("is_success", False):
                if info[0].get("is_success", False):
                    total_success += 1
                break

    vis_env.close() # Important to ensure the video file writes completely
    print(f"success rate: {total_success / n_repeats}")


def main():
    # train(save_path="models/panda_sparse2", total_timesteps=200_000, save_freq=20_000, n_envs=16)
    eval(
        "panda_sparse2",
        10,
        "models/panda_sparse2/checkpoint_200000_steps.zip",
        "models/panda_sparse2/checkpoint_vecnormalize_200000_steps.pkl"
    )


if __name__ == "__main__":
    main()
