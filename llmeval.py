from llm import LLMAgent

with open("sysprompt.txt", "r", encoding="utf-8") as f:
    system_prompt = f.read()

agent = LLMAgent(system_prompt=system_prompt, model="gpt-5", log_path="log.txt", temperature=1.0)

print(agent.generate_response("Analyze the following agent behavior: START, up, up, up, up, END"))
