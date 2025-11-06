#!/usr/bin/env python3
"""
Driver assistance bot that detects driver drowsiness using visual and steering metrics.
Supports multiple model backends (Ollama, OpenAI, Anthropic, etc.).
Strict structured output enforcement using JSON schema.
"""

import toml
import copy
from typing import Optional
from pydantic import BaseModel, Field, StrictFloat
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic


class Bot:
    """Bot schemas"""

    class BotConfig(BaseModel):
        """Configuration for the driver assistance bot."""
        model_id: str
        provider: str = Field(
            default="ollama",
            description="Model provider: 'ollama', 'openai', or 'anthropic'."
        )
        prompt_template: Optional[str] = None
        temperature: Optional[float] = None

    class Input(BaseModel):
        """Input data for the driver assistance bot"""
        window_id: Optional[int] = Field(None, description="Duration of the driving session (in minutes).")
        perclos: Optional[StrictFloat] = Field(None, description="Percentage of time eyes are closed.")
        blink_rate: Optional[StrictFloat] = Field(None, description="Number of eye blinks per minute.")
        blink_duration_mean: Optional[StrictFloat] = Field(None, description="Average duration of eye blinks (seconds).")
        blink_duration_max: Optional[StrictFloat] = Field(None, description="max duration of eye blinks (seconds).")
        yawning_rate: Optional[StrictFloat] = Field(None, description="Number of yawns per minute.")
        sdlp: Optional[StrictFloat] = Field(None, description="Standard deviation of lane position (m).")
        steering_entropy: Optional[StrictFloat] = Field(None, description="Unpredictability measure of steering movements.")
        steering_reversal_rate: Optional[StrictFloat] = Field(None, description="Steering direction changes per minute.")

    # Define JSON schema for structured output
    output_schema = {
        "title": "DriverDrowsinessAssessment",
        "description": "Classification of driver drowsiness level and reasoning based on driving metrics.",
        "type": "object",
        "properties": {
            "drowsiness_level": {
                "type": "integer",
                "description": "Detected drowsiness risk level: 1 = Low, 2 = Moderate, 3 = High"
            },
            "reasoning": {
                "type": "string",
                "description": "Step-by-step reasoning leading to the classification result."
            }
        },
        "required": ["drowsiness_level", "reasoning"]
    }


class BaseBot(Bot):
    """Base class for the driver assistance bot."""

    def __init__(self, config: Bot.BotConfig, enable_history: bool = True, history_limit: int = 10):
        self.config = config
        self.llm = self._get_llm(config)

        # Strict structured output only — always enforce schema
        structured_llm = self.llm.with_structured_output(self.output_schema)
        self.chain = ChatPromptTemplate.from_template(config.prompt_template.strip()) | structured_llm
        self.is_structured = True
        self.history_limit = history_limit
        self.enable_history = enable_history
        self.history = [] 

    def _get_llm(self, config: Bot.BotConfig):
        """Return correct LLM instance based on provider."""
        if config.provider == "ollama":
            return ChatOllama(model=config.model_id, temperature=config.temperature)
        elif config.provider == "openai":
            return ChatOpenAI(model=config.model_id, temperature=config.temperature)
        elif config.provider == "anthropic":
            return ChatAnthropic(model=config.model_id, temperature=config.temperature)
        else:
            raise ValueError(f"Unsupported provider: {config.provider}")
        
    def _update_history(self, user_input: dict, response: dict):
        """Store recent interactions, trimming to last N."""
        self.history.append({"input": user_input, "output": response})
        if len(self.history) > self.history_limit:
            self.history = self.history[-self.history_limit:]


    def invoke(self, input_data: Bot.Input) -> dict:
        """Invoke the driver assistance bot and return structured output (strict JSON)."""
        #  Keep your preferred dict() call
        data = input_data.model_dump()

        raw_data = copy.deepcopy(data)

        if self.enable_history and self.history:
            history_text = "\n".join(
                [
                    f"Turn {i+1}:\n  Input: {h['input']}\n  Output: {h['output']}"
                    for i, h in enumerate(self.history[-self.history_limit:])
                ]
            )
        else:
            history_text = ""

        # Add history into the input variables for the prompt template
        data["history"] = history_text


        response = self.chain.invoke(data)

        #  Must return a valid dict
        if not isinstance(response, dict):
            raise ValueError(
                f" Invalid structured output from model. Expected dict, got {type(response)}.\nResponse: {response}"
            )

        # Validate type of drowsiness_level
        if not isinstance(response.get("drowsiness_level"), int):
            raise ValueError(f" Invalid 'drowsiness_level' type: {type(response.get('drowsiness_level'))}")
        
        if self.enable_history:
            self._update_history(raw_data, response)


        return response


if __name__ == "__main__":
    # Load TOML prompt
    prompt_path = r"/home/karthik/Desktop/drowsiness_detection_project/llm_eval/src/configs/prompt.toml"
    prompt_cfg = toml.load(prompt_path)
    prompt_template = prompt_cfg["driver_prompt"]["prompt"]

    # Bot configuration
    config = Bot.BotConfig(
        provider="ollama",
        model_id="llama3.1:8b",
        prompt_template=prompt_template,
        temperature=0.2,
    )

    bot = BaseBot(config, enable_history = True, history_limit=10)

    # Example input
    for i in range(15):
        input_data = Bot.Input(
            window_id=1 + i,
            perclos=4.5,
            blink_rate=1.0,
            blink_duration_mean=0.2,
            blink_duration_max=0.5,
            yawning_rate=0.0,
            sdlp=0.2,
            steering_entropy=0.1,
            steering_reversal_rate=0.5,
        )
        # Run inference multiple times to test history
        result1 = bot.invoke(input_data)
        print(f"History: {bot.history}")
        print(f"\n--- Structured Output (Run {i + 1}) ---")
        print(result1)
        print("###########################################")

