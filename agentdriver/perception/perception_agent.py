from pathlib import Path
import pickle
import json
import numpy as np

from agentdriver.llm_core.chat import run_one_round_conversation, run_one_round_conversation_with_functional_call
from agentdriver.llm_core.timeout import timeout
from agentdriver.functional_tools.functional_agent import FuncAgent
from agentdriver.perception.perception_prompts import (
    init_system_message,
    detection_prompt,
    prediction_prompt,
    occupancy_prompt,
    map_prompt,
)

class PerceptionAgent:
    def __init__(
        self, 
        token, 
        split, 
        data_path, 
        model_name="gpt-3.5-turbo-0613", 
        verbose=True, 
        backend="openai"
    ) -> None:
        self.token = token
        folder_name = Path("val") if "val" in split else Path("train")
        self.file_name = data_path / folder_name / Path(f"{self.token}.pkl")
        with open(self.file_name, "rb") as f:
            self.data_dict = pickle.load(f)
        self.func_agent = FuncAgent(self.data_dict)
        self.model_name = model_name
        self.verbose = verbose
        self.backend = backend

        self.num_call_detection_times = 1
        self.num_call_prediction_times = 1
        self.num_call_occupancy_times = 1
        self.num_call_map_times = 1

    def functional_call(self, response_message):
        function_name = response_message["function_call"]["name"]
        function_args = json.loads(response_message["function_call"]["arguments"])
        function_to_call = getattr(self.func_agent, function_name)
        if not callable(function_to_call):
            print(f"Function {function_name} is not callable!")
            return None
        else:
            function_returns = function_to_call(**function_args)
            function_prmopt, function_ret_data = function_returns
            if function_prmopt is None:
                function_prmopt = ""
            function_response = {
                "name": function_name,
                "args": function_args,
                "prompt": function_prmopt,
                "data": function_ret_data,
            }
        if self.verbose:
            print(function_name)
            print(function_args)
            print(function_prmopt)
        return function_response

    def generate_detection_func_prompt(self):
        detection_func_prompt = "You can execute one of the following functions to get object detection results (don't execute functions that have been used before):\n"
        for info in self.func_agent.detection_func_infos:
            detection_func_prompt += ("- " + info["name"] + "(")
            if info["parameters"].get("required"):
                pass
            detection_func_prompt += ") #"
            detection_func_prompt += info["description"] + "\n"
        if self.verbose:
            print(detection_func_prompt)
        return detection_func_prompt

    def generate_prediction_func_prompt(self):
        prediction_func_prompt = "You can execute one of the following functions to get object future trajectory predictions (don't execute functions that have been used before):\n"
        for info in self.func_agent.prediction_func_infos:
            prediction_func_prompt += ("- " + info["name"] + "(")
            if info["parameters"].get("required"):
                pass
            prediction_func_prompt += ") #"
            prediction_func_prompt += info["description"] + "\n"
        if self.verbose:
            print(prediction_func_prompt)
        return prediction_func_prompt

    def generate_occupancy_func_prompt(self):
        occupancy_func_prompt = "You can execute one of the following functions to get occupancy information (don't execute functions that have been used before):\n"
        for info in self.func_agent.occupancy_func_infos:
            occupancy_func_prompt += ("- " + info["name"] + "(")
            if info["parameters"].get("required"):
                pass
            occupancy_func_prompt += ") #"
            occupancy_func_prompt += info["description"] + "\n"
        if self.verbose:
            print(occupancy_func_prompt)
        return occupancy_func_prompt

    def generate_map_func_prompt(self):
        map_func_prompt = "You can execute one of the following functions to get map information (don't execute functions that have been used before):\n"
        for info in self.func_agent.map_func_infos:
            map_func_prompt += ("- " + info["name"] + "(")
            if info["parameters"].get("required"):
                pass
            map_func_prompt += ") #"
            map_func_prompt += info["description"] + "\n"
        if self.verbose:
            print(map_func_prompt)
        return map_func_prompt

    def get_perception_results(self, ego_prompts):
        """
        Collecting necessary information from driving scenarios by chain-of-thought reasoning with function call
        """
        full_messages = []
        func_responses = []
        system_message = init_system_message + "\n" + ego_prompts + "\n"

        if self.verbose:
            print(system_message)

        # Detection information
        full_messages, detection_response = run_one_round_conversation(
            full_messages=full_messages, 
            system_message=system_message, 
            user_message=detection_prompt,
            model_name=self.model_name,
            backend=self.backend,
        )

        if self.verbose:
            print(detection_response)

        if detection_response.get("content", "") == "YES":
            # pass
            generate_detection_func_prompt()
        else:
            pass

        # Prediction information
        full_messages, prediction_response = run_one_round_conversation(
            full_messages=full_messages, 
            system_message=None, 
            user_message=prediction_prompt, 
            model_name=self.model_name,
            backend=self.backend,
        )

        if self.verbose:
            print(prediction_response)

        if prediction_response.get("content", "") == "YES":
            # pass
            generate_prediction_func_prompt()
        else:
            pass

        # Occupancy information
        full_messages, occupancy_response = run_one_round_conversation(
            full_messages=full_messages, 
            system_message=None, 
            user_message=occupancy_prompt,
            model_name=self.model_name,
            backend=self.backend,
        )

        if self.verbose:
            print(occupancy_response)

        if occupancy_response.get("content", "") == "YES":
            # pass
            generate_occupancy_func_prompt()
        else:
            pass

        # Map information
        full_messages, map_response = run_one_round_conversation(
            full_messages=full_messages, 
            system_message=None, 
            user_message=map_prompt, 
            model_name=self.model_name,
            backend=self.backend,
        )

        if self.verbose:
            print(map_response)

        if map_response.get("content", "") == "YES":
            # pass
            generate_map_func_prompt()
        else:
            pass

        dummy_ego_data = {
            'ego_states': np.array([1.0]*9),
            'ego_hist_traj_diff': np.array([[0.0, 0.0], [1.0, 1.0]]),
            'ego_hist_traj': np.random.rand(5, 2),
            'goal': np.array([0.0, 0.0])
        }

        working_memory = self.process_perception_results(ego_prompts, dummy_ego_data, full_messages, func_responses)

        # For demonstration, return dummy values
        return ego_prompts, "perception_prompts", working_memory

    def process_perception_results(self, ego_prompts, ego_data, full_messages, func_responses):
        perception_prompts = "*"*5 + "Perception Results:" + "*"*5 + "\n"
        working_memory = {}
        working_memory["token"] = self.token
        working_memory["ego_data"] = ego_data
        working_memory["functions"] = {}
        for func_response in func_responses:
            pass
        if self.verbose:
            pass
        working_memory.update({"perception_prompts": perception_prompts})
        working_memory.update({"ego_prompts": ego_prompts})
        return working_memory

    @timeout(60)
    def run(self):
        # You may want to implement the actual logic here
        ego_prompts = "ego_prompts"
        return self.get_perception_results(ego_prompts)
