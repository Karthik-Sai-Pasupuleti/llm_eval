#!/usr/bin/env python3
"""
Isolated script to preview center crop on a single frame.
Run this to tune crop parameters before running the full VLM pipeline.
"""

import cv2
import numpy as np


VIDEO_PATH = r"/home/vanchha/Refined_Participants_Data/08_D_Data/videos/window_46.mp4"
FRAME_INDEX = 0

# Crop parameters — tune these
CROP_WIDTH_FRACTION = 0.5   # fraction of original width to keep
CROP_HEIGHT_FRACTION = 0.75  # fraction of original height to keep
OFFSET_X = 0           # shift crop left(-) or right(+) in pixels
OFFSET_Y = 100                # shift crop up(-) or down(+) in pixels


def extract_single_frame(video_path: str, frame_index: int = 0) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Could not read frame {frame_index}")
    return frame


def crop_frame(
    frame: np.ndarray,
    width_fraction: float = 0.5,
    height_fraction: float = 0.5,
    offset_x: int = 0,
    offset_y: int = 0,
) -> np.ndarray:
    h, w = frame.shape[:2]
    new_w = int(w * width_fraction)
    new_h = int(h * height_fraction)

    # Center position with offset
    left = max(0, min((w - new_w) // 2 + offset_x, w - new_w))
    top = max(0, min((h - new_h) // 2 + offset_y, h - new_h))

    return frame[top:top + new_h, left:left + new_w]


def show_comparison(original: np.ndarray, cropped: np.ndarray):
    # Resize both to same height for side-by-side display
    target_h = 480
    scale_orig = target_h / original.shape[0]
    scale_crop = target_h / cropped.shape[0]

    orig_display = cv2.resize(original, (int(original.shape[1] * scale_orig), target_h))
    crop_display = cv2.resize(cropped, (int(cropped.shape[1] * scale_crop), target_h))

    # Add labels
    cv2.putText(orig_display, "ORIGINAL", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(crop_display, "CROPPED", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    combined = np.hstack([orig_display, crop_display])
    cv2.imshow("Crop Preview — press any key to close", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Also save to disk
    cv2.imwrite("crop_preview.jpg", combined)
    print("Saved preview to crop_preview.jpg")


if __name__ == "__main__":
    frame = extract_single_frame(VIDEO_PATH, FRAME_INDEX)
    print(f"Original size: {frame.shape[1]}x{frame.shape[0]} (WxH)")

    cropped = crop_frame(
        frame,
        width_fraction=CROP_WIDTH_FRACTION,
        height_fraction=CROP_HEIGHT_FRACTION,
        offset_x=OFFSET_X,
        offset_y=OFFSET_Y,
    )
    print(f"Cropped size:  {cropped.shape[1]}x{cropped.shape[0]} (WxH)")
    print(f"Parameters used: width_fraction={CROP_WIDTH_FRACTION}, height_fraction={CROP_HEIGHT_FRACTION}, offset_x={OFFSET_X}, offset_y={OFFSET_Y}")

    show_comparison(frame, cropped)
