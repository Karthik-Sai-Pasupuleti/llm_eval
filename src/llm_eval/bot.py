#!/usr/bin/env python3
"""
Driver assistance bot that detects driver drowsiness using visual and steering metrics.
Supports multiple model backends (Ollama, OpenAI, Anthropic, etc.).
Strict structured output enforcement using JSON schema.
"""

import toml
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
        blink_duration: Optional[StrictFloat] = Field(None, description="Average duration of eye blinks (seconds).")
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

    def __init__(self, config: Bot.BotConfig):
        self.config = config
        self.llm = self._get_llm(config)

        # Strict structured output only — always enforce schema
        structured_llm = self.llm.with_structured_output(self.output_schema)
        self.chain = ChatPromptTemplate.from_template(config.prompt_template.strip()) | structured_llm
        self.is_structured = True

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

    def invoke(self, input_data: Bot.Input) -> dict:
        """Invoke the driver assistance bot and return structured output (strict JSON)."""
        #  Keep your preferred dict() call
        data = input_data.model_dump()

        response = self.chain.invoke(data)

        #  Must return a valid dict
        if not isinstance(response, dict):
            raise ValueError(
                f" Invalid structured output from model. Expected dict, got {type(response)}.\nResponse: {response}"
            )

        # Validate type of drowsiness_level
        if not isinstance(response.get("drowsiness_level"), int):
            raise ValueError(f" Invalid 'drowsiness_level' type: {type(response.get('drowsiness_level'))}")

        return response


if __name__ == "__main__":
    # Load TOML prompt
    prompt_path = r"C:\Users\pasupuleti\Desktop\group-project\experiments\llm_eval\src\configs\prompt.toml"
    prompt_cfg = toml.load(prompt_path)
    prompt_template = prompt_cfg["driver_prompt"]["prompt"]

    # Bot configuration
    config = Bot.BotConfig(
        provider="ollama",
        model_id="llama3.1:8b",
        prompt_template=prompt_template,
        temperature=0.2,
    )

    bot = BaseBot(config)

    # Example input
    input_data = Bot.Input(
        window_id=1,
        perclos=4.5,
        blink_rate=1.0,
        blink_duration=0.2,
        yawning_rate=0.0,
        sdlp=0.,
        steering_entropy=0.,
        steering_reversal_rate=0.0,
    )

    # Run inference
    result = bot.invoke(input_data)

    print("\n Structured Output:")
    print(result)
