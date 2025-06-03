import os
import requests

class LLMBackend:
    def __init__(self, backend="openai", model_name="gpt-3.5-turbo-0613"):
        self.backend = backend
        self.model_name = model_name

    def chat(self, messages, temperature=0.0, functions=None, function_call=None):
        if self.backend == "openai":
            import openai
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                functions=functions,
                function_call=function_call,
            )
            return response
        elif self.backend == "ollama":
            # Ollama API expects a different format
            url = "http://localhost:11434/api/chat"
            payload = {
                "model": self.model_name,
                "messages": messages,
                "options": {"temperature": temperature},
            }
            if functions:
                payload["functions"] = functions
            if function_call:
                payload["function_call"] = function_call
            resp = requests.post(url, json=payload)
            resp.raise_for_status()
            return resp.json()
        else:
            raise ValueError(f"Unknown backend: {self.backend}")