#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass

from ..configs import CameraConfig, ColorMode, Cv2Rotation


@CameraConfig.register_subclass("gemini")
@dataclass(kw_only=True)
class GeminiCameraConfig(CameraConfig):
    """Configuration for Orbbec Gemini camera using pyorbbecsdk.

    Attributes:
        serial_number_or_name: Camera serial number or unique device name.
        fps: Frames per second. If None, uses device default.
        width: Frame width in pixels. If None, uses device default.
        height: Frame height in pixels. If None, uses device default.
        color_mode: Output color format (RGB or BGR). Defaults to RGB.
        rotation: Image rotation to apply. Defaults to NO_ROTATION.
        use_depth: Whether to enable depth stream. Defaults to False.
        warmup_s: Warmup time in seconds before first read. Defaults to 1.0.
    """

    serial_number_or_name: str = ""
    color_mode: ColorMode = ColorMode.RGB
    rotation: Cv2Rotation = Cv2Rotation.NO_ROTATION
    use_depth: bool = False
    warmup_s: float = 1.0
