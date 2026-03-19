#!/usr/bin/env python3
"""
VLM bot for driver drowsiness detection using visual frame analysis.
Accepts pre-cropped video frames, builds a contact sheet, and queries a VLM via Ollama.
Returns structured output with ambiguity and facial behaviour detections.
"""

import re
import cv2
import json
import base64
import numpy as np
from typing import Optional
from pydantic import BaseModel, Field
from ollama import Client as OllamaClient


OLLAMA_HOST = "http://localhost:11434"

VLM_PROMPT = """You are analyzing a contact sheet of 60 frames (arranged in a grid, left to right, top to bottom) from a 60-second driver-facing camera window. Each cell is one second apart.

Your task is to detect the following ONLY if clearly present:
1. Visual ambiguities: occlusions, sunglasses, hand covering mouth, hand covering face, poor lighting, motion blur, partial face obstruction.
2. Facial behaviour: yawning, head nodding, head tilting to one side, eyes closing, microsleep, prolonged eye closure.

Each frame is labelled with its frame_id in the top-left corner. When reporting detections, include the frame_ids where they occur.

Respond strictly in this JSON format:
{
  "ambiguities_detected": true or false,
  "ambiguity_frame_ids": [list of integer frame_ids where ambiguities are present, empty if none],
  "ambiguity_types": ["list of detected ambiguity types, empty if none"],
  "facial_behaviour_detected": true or false,
  "facial_behaviour_frame_ids": [list of integer frame_ids where facial behaviours are present, empty if none],
  "facial_behaviours": ["list of detected facial behaviours, empty if none"],
  "description": "brief one or two sentence summary of what is observed overall",
  "confidence": "low, medium, or high"
}

Only report what is clearly visible. Do not hallucinate. If nothing detected, set both flags to false and use empty lists."""


class VLMBot:
    """VLM bot schemas and configuration."""

    class BotConfig(BaseModel):
        model_id: str
        temperature: float = Field(default=0.1)
        ollama_host: str = Field(default=OLLAMA_HOST)
        contact_sheet_cols: int = Field(default=10)
        thumb_width: int = Field(default=160)

    class Input(BaseModel):
        window_id: Optional[str] = Field(None)
        video_path: Optional[str] = Field(None)
        frames: Optional[list] = Field(None, description="Pre-loaded frames as numpy arrays. If provided, video_path is ignored.")
        drowsiness_level: Optional[int] = Field(None)

    class Output(BaseModel):
        window_id: Optional[str] = None
        ambiguities_detected: Optional[bool] = None
        ambiguity_frame_ids: list[int] = []
        ambiguity_types: list[str] = []
        facial_behaviour_detected: Optional[bool] = None
        facial_behaviour_frame_ids: list[int] = []
        facial_behaviours: list[str] = []
        description: Optional[str] = None
        confidence: Optional[str] = None
        parse_error: bool = False
        error: Optional[str] = None


class BaseVLMBot(VLMBot):
    """Base class for VLM-based driver drowsiness visual analysis."""

    def __init__(self, config: VLMBot.BotConfig):
        self.config = config
        self.client = OllamaClient(host=config.ollama_host)

    def _extract_frames(self, video_path: str, num_frames: int = 60) -> list[np.ndarray]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            raise RuntimeError(f"Video has 0 frames: {video_path}")

        indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        cap.release()
        return frames

    def _build_contact_sheet(self, frames: list[np.ndarray]) -> np.ndarray:
        cols = self.config.contact_sheet_cols
        thumb_width = self.config.thumb_width

        thumbs = []
        for frame_id, f in enumerate(frames):
            h, w = f.shape[:2]
            scale = thumb_width / w
            thumb = cv2.resize(f, (thumb_width, int(h * scale)))
            cv2.putText(
                thumb, str(frame_id), (4, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA
            )
            thumbs.append(thumb)

        rows = (len(thumbs) + cols - 1) // cols
        thumb_h, thumb_w = thumbs[0].shape[:2]
        blank = np.zeros((thumb_h, thumb_w, 3), dtype=np.uint8)
        while len(thumbs) % cols != 0:
            thumbs.append(blank)

        grid_rows = [np.hstack(thumbs[r * cols:(r + 1) * cols]) for r in range(rows)]
        return np.vstack(grid_rows)

    def _frame_to_base64(self, frame: np.ndarray) -> str:
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return base64.b64encode(buffer).decode("utf-8")

    def _parse_response(self, raw: str) -> dict:
        try:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception:
            pass
        return {
            "ambiguities_detected": None,
            "ambiguity_frame_ids": [],
            "ambiguity_types": [],
            "facial_behaviour_detected": None,
            "facial_behaviour_frame_ids": [],
            "facial_behaviours": [],
            "description": raw.strip(),
            "confidence": "low",
            "parse_error": True,
        }

    def invoke(self, input_data: VLMBot.Input, num_frames: int = 60) -> VLMBot.Output:
        """
        Invoke the VLM bot on a single window.
        Accepts either pre-loaded frames or a video_path to extract from.
        """
        try:
            if input_data.frames is not None:
                frames = input_data.frames
            elif input_data.video_path is not None:
                frames = self._extract_frames(input_data.video_path, num_frames=num_frames)
            else:
                raise ValueError("Either frames or video_path must be provided.")

            if not frames:
                raise RuntimeError("No frames available.")

            collage = self._build_contact_sheet(frames)
            collage_b64 = self._frame_to_base64(collage)

            response = self.client.chat(
                model=self.config.model_id,
                options={"temperature": self.config.temperature},
                messages=[
                    {
                        "role": "user",
                        "content": VLM_PROMPT,
                        "images": [collage_b64],
                    }
                ],
            )
            raw = response["message"]["content"]
            parsed = self._parse_response(raw)
            parsed["window_id"] = input_data.window_id
            parsed["error"] = None

            return VLMBot.Output(**{
                k: v for k, v in parsed.items()
                if k in VLMBot.Output.model_fields
            })

        except Exception as e:
            return VLMBot.Output(
                window_id=input_data.window_id,
                error=str(e),
            )


if __name__ == "__main__":
    config = VLMBot.BotConfig(
        model_id="llava:13b",
        temperature=0.1,
        contact_sheet_cols=10,
        thumb_width=160,
    )

    bot = BaseVLMBot(config)

    input_data = VLMBot.Input(
        window_id="30",
        video_path="/home/vanchha/llm_eval/Cropped_Videos/04_SB/window_30.mp4",
    )

    result = bot.invoke(input_data)
    print(result.model_dump_json(indent=2))
