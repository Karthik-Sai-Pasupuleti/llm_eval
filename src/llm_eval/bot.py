#!/usr/bin/env python3
"""
Driver assistance bot that detects driver drowsiness using visual, steering, and physiological metrics.
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


def _fmt(value, spec: str) -> str:
    """Format a numeric value with the given format spec, return 'N/A' if None or NaN."""
    if value is None:
        return "N/A"
    try:
        import math
        if math.isnan(float(value)):
            return "N/A"
        return format(float(value), spec)
    except Exception:
        return "N/A"


class Bot:
    """Bot schemas"""

    class BotConfig(BaseModel):
        model_id: str
        provider: str = Field(default="ollama")
        prompt_template: Optional[str] = None
        temperature: Optional[float] = None

    class Input(BaseModel):
        window_id: Optional[int] = Field(None)
        perclos: Optional[StrictFloat] = Field(None)
        blink_rate: Optional[StrictFloat] = Field(None)
        blink_duration_mean: Optional[StrictFloat] = Field(None)
        blink_duration_max: Optional[StrictFloat] = Field(None)
        yawning_rate: Optional[StrictFloat] = Field(None)
        sdlp: Optional[StrictFloat] = Field(None)
        steering_entropy: Optional[StrictFloat] = Field(None)
        steering_reversal_rate: Optional[StrictFloat] = Field(None)
        bpm: Optional[StrictFloat] = Field(None)
        hrv_sdnn: Optional[StrictFloat] = Field(None)
        hrv_rmssd: Optional[StrictFloat] = Field(None)
        hrv_sd1: Optional[StrictFloat] = Field(None)
        hrv_hf: Optional[StrictFloat] = Field(None)
        hrv_wavelet_entropy: Optional[StrictFloat] = Field(None)
        hrv_lfhf: Optional[StrictFloat] = Field(None)

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

    def __init__(self, config: Bot.BotConfig, enable_history: bool = True, history_limit: int = 10):
        self.config = config
        self.llm = self._get_llm(config)
        structured_llm = self.llm.with_structured_output(self.output_schema)
        self.chain = ChatPromptTemplate.from_template(config.prompt_template.strip()) | structured_llm
        self.is_structured = True
        self.history_limit = history_limit
        self.enable_history = enable_history
        self.history = []

    def _get_llm(self, config: Bot.BotConfig):
        if config.provider == "ollama":
            return ChatOllama(model=config.model_id, temperature=config.temperature)
        elif config.provider == "openai":
            return ChatOpenAI(model=config.model_id, temperature=config.temperature)
        elif config.provider == "anthropic":
            return ChatAnthropic(model=config.model_id, temperature=config.temperature)
        else:
            raise ValueError(f"Unsupported provider: {config.provider}")

    def _update_history(self, user_input: dict, response: dict):
        self.history.append({"input": user_input, "output": response})
        if len(self.history) > self.history_limit:
            self.history = self.history[-self.history_limit:]

    def invoke(self, input_data: Bot.Input) -> dict:
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

        # Pre-format all numeric fields as strings to avoid NoneType format errors
        prompt_data = {
            "window_id": data.get("window_id") if data.get("window_id") is not None else "N/A",
            "perclos": _fmt(data.get("perclos"), ".2f"),
            "blink_rate": _fmt(data.get("blink_rate"), ".1f"),
            "blink_duration_mean": _fmt(data.get("blink_duration_mean"), ".3f"),
            "blink_duration_max": _fmt(data.get("blink_duration_max"), ".3f"),
            "yawning_rate": _fmt(data.get("yawning_rate"), ".1f"),
            "sdlp": _fmt(data.get("sdlp"), ".2f"),
            "steering_entropy": _fmt(data.get("steering_entropy"), ".2f"),
            "steering_reversal_rate": _fmt(data.get("steering_reversal_rate"), ".2f"),
            "bpm": _fmt(data.get("bpm"), ".1f"),
            "hrv_sdnn": _fmt(data.get("hrv_sdnn"), ".2f"),
            "hrv_rmssd": _fmt(data.get("hrv_rmssd"), ".2f"),
            "hrv_sd1": _fmt(data.get("hrv_sd1"), ".2f"),
            "hrv_hf": _fmt(data.get("hrv_hf"), ".4f"),
            "hrv_wavelet_entropy": _fmt(data.get("hrv_wavelet_entropy"), ".4f"),
            "hrv_lfhf": _fmt(data.get("hrv_lfhf"), ".3f"),
            "history": history_text,
        }

        response = self.chain.invoke(prompt_data)

        if not isinstance(response, dict):
            raise ValueError(
                f"Invalid structured output from model. Expected dict, got {type(response)}.\nResponse: {response}"
            )

        if not isinstance(response.get("drowsiness_level"), int):
            raise ValueError(f"Invalid 'drowsiness_level' type: {type(response.get('drowsiness_level'))}")

        if self.enable_history:
            self._update_history(raw_data, response)

        return response
