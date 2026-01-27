import re

import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env

from llm import LLMAgent, format_video_input
from train_agent import AgentTrainer, visualize_agent, MiniWrapper


def create_wrapper(video_path: str, behavior_agent: LLMAgent, reward_agent: LLMAgent, frame_skip: int = 10) -> str:
    """
    Takes in a video path of a behavior and outputs a wrapper as prescribed by the LLM agents.
    """

    video_input = format_video_input(video_path, frame_skip=frame_skip)

    behavior_response = behavior_agent.generate_response(video_input)
    print("Behavior Agent Response:")
    print(behavior_response)

    reward_response = reward_agent.generate_response(behavior_response)
    print("Reward Agent Response:")
    print(reward_response)

    # Get the last python block from the response
    code_blocks = re.findall(r"```python(.*?)```", reward_response, re.DOTALL)
    if code_blocks:
        step_fn = code_blocks[-1].strip()
    else:
        raise ValueError("No python code block found in the reward agent response.")

    return step_fn


def get_wrapper(wrapper_code: str) -> gym.Wrapper:
    """
    Given wrapper code as a string, return the Wrapper class defined
    """
    namespace = {}
    exec(wrapper_code, namespace)
    return namespace["DoorKeyRewardWrapper"]


def main():
    """
    Main logic for experiment
    """
    # Initialize agents and trainer
    # with open("behavior.txt", "r", encoding="utf-8") as f:
    #     behavior_prompt = f.read()
    # behavior_agent = LLMAgent(behavior_prompt, model="gpt-5.2", temperature=1.0, log_path="logs/behavior_agent.log")

    # with open("rewarder.txt", "r", encoding="utf-8") as f:
    #     rewarder_prompt = f.read()
    # reward_agent = LLMAgent(rewarder_prompt, model="gpt-5.2", temperature=1.0, log_path="logs/reward_agent.log")

    # wrapper_code = create_wrapper(
    #     video_path="videos/spin.mp4",
    #     behavior_agent=behavior_agent,
    #     reward_agent=reward_agent,
    #     frame_skip=10
    # )

    with open("temp.py", "r", encoding="utf-8") as f:
        wrapper_code = f.read()

    new_wrapper = get_wrapper(wrapper_code)
    env = make_vec_env(
        "MiniGrid-DoorKey-5x5-v0",
        n_envs=8,
        seed=42,
        wrapper_class=MiniWrapper,
        wrapper_kwargs={"custom_wrapper": new_wrapper}
    )

    trainer = AgentTrainer(seed=42, device="cpu")
    agent = trainer.train_agent(env, steps=500_000, verbose=1)

    agent.save("models/modified")

    vis_env = gym.make("MiniGrid-DoorKey-5x5-v0", render_mode="rgb_array")
    vis_env = MiniWrapper(vis_env, custom_wrapper=new_wrapper)
    visualize_agent(agent, vis_env, save_prefix="minigrid_doorkey_agent_modified")


if __name__ == "__main__":
    main()
