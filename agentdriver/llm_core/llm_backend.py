import os
import requests
import json

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

            resp = requests.post(url, json=payload, stream=True)
            resp.raise_for_status()
            
            # Ollama streams JSON objects, one per line
            response_content = ""
            last_data = None
            for line in resp.iter_lines():
                if line:
                    data = json.loads(line.decode("utf-8"))
                    last_data = data
                    # Some chunks contain part of the assistant's message
                    if "message" in data and "content" in data["message"]:
                        response_content += data["message"]["content"]
            if last_data is None:
                raise RuntimeError("No response from Ollama.")
            return response_content
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
