from cerebrum.agents.base import BaseAgent
from cerebrum.llm.communication import LLMQuery
import json

class MyAgent(BaseAgent):
    def __init__(self, agent_name, task_input, config_):
        super().__init__(agent_name, task_input, config_)

        # AIOS can automatically run the agent multiple times for you!
        self.plan_max_fail_times = 3
        self.tool_call_max_fail_times = 3

        self.task_input = task_input
        self.messages = []
        self.workflow_mode = "manual"  # (manual, automatic)
        self.rounds = 0

    def manual_workflow(self):
        # turned this simple library into an agent
        workflow = [
            {
                "action_type": "tool_use",
                "action": "Search for relevant papers",
                "tool_use": ["example/arxiv"],
            },
            {
                "action_type": "chat",
                "action": "Provide responses based on the user's query",
                "tool_use": [],
            },
        ]
        return workflow

    def run(self):
        # manual workflow
        self.messages.append({"role": "system", "content": 
              "".join(["".join(self.config["description"])])})

        task_input = self.task_input

        self.messages.append({"role": "user", "content": task_input})

        workflow = self.manual_workflow()

        # chain of thought 
        self.messages.append(
            {
                "role": "user",
                "content": f"[Thinking]: The workflow for the problem is {json.dumps(workflow)}. Follow the workflow to solve the problem step by step. ",
            }
        )

        final_result = ""

        # scalable workflow generator 
        for i, step in enumerate(workflow):
            action_type = step["action_type"]
            action = step["action"]
            tool_use = step["tool_use"]

            prompt = f"At step {i + 1}, you need to: {action}. "
            self.messages.append({"role": "user", "content": prompt})

            if tool_use:
                selected_tools = self.pre_select_tools(tool_use)

            else:
                selected_tools = None

            response = self.send_request(
                agent_name=self.agent_name,
                query=LLMQuery(
                    messages=self.messages,
                    tools=selected_tools,
                    action_type=action_type,
                ),
            )["response"]
            
            self.messages.append({"role": "assistant", "content": response.response_message})

            self.rounds += 1


        final_result = self.messages[-1]["content"]
                
        return {
            "agent_name": self.agent_name,
            "result": final_result,
            "rounds": self.rounds,
        }

