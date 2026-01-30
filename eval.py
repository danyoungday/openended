from pathlib import Path

import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from get_help import get_wrapper
from train_agent import AgentTrainer, MiniWrapper

def train_method(save_path: str, wrapper_path: str) -> PPO:
    """
    Trains an agent with an optional custom wrapper and saves the model. Validates on the same environment as training
    but without the custom wrapper around it.
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=False)

    trainer = AgentTrainer(seed=42, device="cpu", log_path=str(save_path / "logs"))

    # Set up environment with custom wrapper if provided
    wrapper_kwargs = {}
    if wrapper_path is not None:
        print("Wrapping env")
        with open(wrapper_path, "r", encoding="utf-8") as f:
            wrapper_code = f.read()
        new_wrapper = get_wrapper(wrapper_code)
        print(new_wrapper)
        wrapper_kwargs = {"wrapper_kwargs": {"custom_wrapper": new_wrapper}}

    # Use the custom wrapper in the vectorized environment if provided
    env = make_vec_env(
        "MiniGrid-DoorKey-5x5-v0",
        n_envs=8,
        seed=42,
        monitor_dir=str(save_path / "monitor"),
        wrapper_class=MiniWrapper,
        **wrapper_kwargs
    )

    agent = trainer.train_agent(
        env,
        steps=200_000,
        verbose=1
    )
    agent.save(save_path / "final")

    return agent


def eval_agent(agent: PPO) -> float:
    """
    Evals a trained agent on the set environment without the custom wrapper
    """
    trainer = AgentTrainer(seed=42, device="cpu")
    agent = PPO.load("models/baseline/final.zip")
    trainer.evaluate_agent(
        agent,
        MiniWrapper(gym.make("MiniGrid-DoorKey-5x5-v0", render_mode="rgb_array")),
        n_episodes=10,
        record_name="baseline_eval"
    )

if __name__ == "__main__":
    train_method("models/modified", "temp.py")
