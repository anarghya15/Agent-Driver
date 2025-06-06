# Basic Chat Completion Functions
# Written by Jiageng Mao 

import json
from typing import List, Dict
from agentdriver.llm_core.chat_utils import completion_with_backoff
from agentdriver.llm_core.timeout import timeout
from agentdriver.llm_core.llm_backend import LLMBackend


@timeout(60)
def run_one_round_conversation(
        full_messages: List[Dict], 
        system_message: str, 
        user_message: str,
        temperature: float = 0.0,
        model_name: str = "gpt-3.5-turbo-0613", # "gpt-3.5-turbo-16k-0613"
        backend="openai"
    ):
    """
    Perform one round of conversation using OpenAI API
    """
    # message_for_this_round = [
    #         {"role": "system", "content": system_message},
    #         {"role": "user", "content": user_message},
    #     ] if system_message else [{"role": "user", "content": user_message}]
    
    # full_messages.extend(message_for_this_round)
    
    # response = completion_with_backoff(
    #     model=model_name,
    #     messages=full_messages,
    #     temperature=temperature,
    # )

    # response_message = response["choices"][0]["message"]
    
    # # Append assistant's reply to conversation
    # full_messages.append(response_message)

    # return full_messages, response_message
    message_for_this_round = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ] if system_message else [{"role": "user", "content": user_message}]
    
    full_messages.extend(message_for_this_round)
    
    llm = LLMBackend(backend=backend, model_name=model_name)
    
    response = llm.chat(messages=full_messages, temperature=temperature)
    if backend == "openai":
        response_message = response["choices"][0]["message"]
    elif backend == "ollama":
        # Wrap as a message dict, so the rest of your code expects the same format
        response_message = {
            "role": "assistant",
            "content": response
        }
    else:
        raise ValueError(f"Unknown backend: {backend}")
    
    full_messages.append(response_message)
    
    return full_messages, response_message

def run_one_round_conversation_with_functional_call(
        full_messages: List[Dict], 
        system_message: str, 
        user_message: str, 
        functional_calls_info: List[Dict],
        temperature: float = 0.0,
        model_name: str = "gpt-3.5-turbo-0613", # "gpt-3.5-turbo-16k-0613"
        backend="openai"
    ):
    """
    Perform one round of conversation with functional call using OpenAI API
    """
    # message_for_this_round = [
    #         {"role": "system", "content": system_message},
    #         {"role": "user", "content": user_message},
    #     ] if system_message else [{"role": "user", "content": user_message}]
    
    # full_messages.extend(message_for_this_round)
    
    # response = completion_with_backoff(
    #     model=model_name,
    #     messages=full_messages,
    #     temperature=temperature,
    #     functions=functional_calls_info,
    #     function_call="auto",
    # )

    # response_message = response["choices"][0]["message"]
    
    # # Append assistant's reply to conversation
    # full_messages.append(response_message)
    
    # return full_messages, response_message

    message_for_this_round = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ] if system_message else [{"role": "user", "content": user_message}]

    full_messages.extend(message_for_this_round)
    
    llm = LLMBackend(backend=backend, model_name=model_name)
    
    response = llm.chat(
        messages=full_messages,
        temperature=temperature,
        functions=functional_calls_info,
        function_call="auto",
    )
    
    response_message = response["choices"][0]["message"] if backend == "openai" else response["message"]
    
    full_messages.append(response_message)
    
    return full_messages, response_message
