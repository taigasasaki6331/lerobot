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

"""
Provides the GeminiCamera class for capturing frames from Orbbec Gemini cameras.
"""

import logging
import time
from threading import Event, Lock, Thread
from typing import Any

import cv2  # type: ignore  # TODO: add type stubs for OpenCV
import numpy as np  # type: ignore  # TODO: add type stubs for numpy
from numpy.typing import NDArray  # type: ignore  # TODO: add type stubs for numpy.typing

try:
    from pyorbbecsdk import Context, Pipeline, Config, OBSensorType, OBFormat, OBStreamType
except Exception as e:
    logging.info(f"Could not import pyorbbecsdk: {e}")

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..camera import Camera
from ..configs import ColorMode
from ..utils import get_cv2_rotation
from .configration_gemini import GeminiCameraConfig

logger = logging.getLogger(__name__)


class GeminiCamera(Camera):
    """
    Manages interactions with Orbbec Gemini cameras for frame and depth recording.

    This class provides an interface similar to RealSenseCamera but for Orbbec devices,
    leveraging the pyorbbecsdk library. It uses the camera's unique serial number for
    identification, offering more stability than device indices. It also supports
    capturing depth maps alongside color frames.

    Use the provided utility script to find available camera indices and default profiles:
    ```bash
    lerobot-find-cameras gemini
    ```

    A GeminiCamera instance requires a configuration object specifying the camera's
    serial number or a unique device name. If using the name, ensure only one camera
    with that name is connected.

    Example:
        ```python
        from lerobot.cameras.gemini import GeminiCamera, GeminiCameraConfig
        from lerobot.cameras import ColorMode, Cv2Rotation

        # Basic usage with serial number
        config = GeminiCameraConfig(serial_number_or_name="ABC123456789")
        camera = GeminiCamera(config)
        camera.connect()

        # Read 1 frame synchronously
        color_image = camera.read()
        print(color_image.shape)

        # Read 1 frame asynchronously
        async_image = camera.async_read()

        # When done, properly disconnect the camera
        camera.disconnect()

        # Example with depth capture and custom settings
        custom_config = GeminiCameraConfig(
            serial_number_or_name="ABC123456789",
            fps=30,
            width=640,
            height=480,
            color_mode=ColorMode.BGR,
            rotation=Cv2Rotation.NO_ROTATION,
            use_depth=True
        )
        depth_camera = GeminiCamera(custom_config)
        depth_camera.connect()

        # Read 1 depth frame
        depth_map = depth_camera.read_depth()
        ```
    """

    def __init__(self, config: GeminiCameraConfig):
        """
        Initializes the GeminiCamera instance.

        Args:
            config: The configuration settings for the camera.
        """
        super().__init__(config)

        self.config = config

        # Try to determine if it's a serial number or name
        # Serial numbers typically contain alphanumeric characters
        # Names usually have spaces or are more descriptive
        if " " in config.serial_number_or_name or len(config.serial_number_or_name) < 6:
            # Looks like a name, try to find the serial number
            self.serial_number = self._find_serial_number_from_name(config.serial_number_or_name)
        else:
            # Assume it's a serial number
            self.serial_number = config.serial_number_or_name

        self.fps = config.fps
        self.color_mode = config.color_mode
        self.use_depth = config.use_depth
        self.warmup_s = config.warmup_s

        self.pipeline: Pipeline | None = None
        self.pipe_config: Config | None = None

        self.thread: Thread | None = None
        self.stop_event: Event | None = None
        self.frame_lock: Lock = Lock()
        self.latest_frame: NDArray[Any] | None = None
        self.new_frame_event: Event = Event()

        self.rotation: int | None = get_cv2_rotation(config.rotation)

        if self.height and self.width:
            self.capture_width, self.capture_height = self.width, self.height
            if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                self.capture_width, self.capture_height = self.height, self.width

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.serial_number})"

    @property
    def is_connected(self) -> bool:
        """Checks if the camera pipeline is started and streams are active."""
        return self.pipeline is not None

    def connect(self, warmup: bool = True) -> None:
        """
        Connects to the Orbbec Gemini camera specified in the configuration.

        Initializes the Orbbec pipeline, configures the required streams (color
        and optionally depth), starts the pipeline, and validates the actual stream settings.

        Raises:
            DeviceAlreadyConnectedError: If the camera is already connected.
            ValueError: If the configuration is invalid.
            ConnectionError: If the camera fails to start the pipeline.
            RuntimeError: If the pipeline starts but fails to apply requested settings.
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} is already connected.")

        try:
            ctx = Context()
            device_list = ctx.query_devices()

            # Find device by serial number
            device = None
            for i in range(device_list.get_count()):
                dev = device_list.get_device_by_index(i)
                dev_info = dev.get_device_info()
                if dev_info.get_serial_number() == self.serial_number:
                    device = dev
                    break

            if device is None:
                raise ConnectionError(
                    f"Failed to find {self} with serial number {self.serial_number}. "
                    f"Run `lerobot-find-cameras gemini` to find available cameras."
                )

            self.pipeline = Pipeline(device)
            self.pipe_config = Config()
            self._configure_pipeline_config()

            self.pipeline.start(self.pipe_config)

        except Exception as e:
            self.pipeline = None
            self.pipe_config = None
            raise ConnectionError(
                f"Failed to open {self}. Run `lerobot-find-cameras gemini` to find available cameras."
            ) from e

        # Wait a bit before reading config to let camera stabilize
        # Especially important when both color and depth streams are enabled
        time.sleep(2 if self.use_depth else 1)

        self._configure_capture_settings()

        if warmup:
            start_time = time.time()
            while time.time() - start_time < self.warmup_s:
                try:
                    self.read(timeout_ms=3000)
                    time.sleep(0.1)
                except Exception as e:
                    logger.debug(f"Warmup read failed: {e}")
                    time.sleep(0.2)

        logger.info(f"{self} connected.")

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        """
        Detects available Orbbec Gemini cameras connected to the system.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries,
            where each dictionary contains 'type', 'id' (serial number), 'name',
            firmware version, and other available specs.

        Raises:
            OSError: If pyorbbecsdk is not installed.
            ImportError: If pyorbbecsdk is not installed.
        """
        found_cameras_info = []

        try:
            ctx = Context()
            device_list = ctx.query_devices()

            for i in range(device_list.get_count()):
                device = device_list.get_device_by_index(i)
                device_info = device.get_device_info()

                camera_info = {
                    "name": device_info.get_name(),
                    "type": "Gemini",
                    "id": device_info.get_serial_number(),
                    "firmware_version": device_info.get_firmware_version(),
                    "hardware_version": device_info.get_hardware_version(),
                    "connection_type": device_info.get_connection_type(),
                }

                # Get available stream profiles (video only)
                sensor_list = device.get_sensor_list()
                for j in range(sensor_list.get_count()):
                    sensor = sensor_list.get_sensor_by_index(j)
                    profiles = sensor.get_stream_profile_list()

                    # Get the default video stream profile
                    try:
                        profile = profiles.get_default_video_stream_profile()
                        if profile is not None:
                            stream_info = {
                                "stream_type": str(profile.get_type()),
                                "format": str(profile.get_format()),
                                "width": profile.get_width(),
                                "height": profile.get_height(),
                                "fps": profile.get_fps(),
                            }
                            camera_info["default_stream_profile"] = stream_info
                            break  # Found default profile, no need to continue
                    except Exception:
                        # This sensor may not have video profiles (e.g., IMU)
                        continue

                found_cameras_info.append(camera_info)

        except Exception as e:
            logger.error(f"Error finding Gemini cameras: {e}")

        return found_cameras_info

    def _find_serial_number_from_name(self, name: str) -> str:
        """Finds the serial number for a given unique camera name."""
        camera_infos = self.find_cameras()
        found_devices = [cam for cam in camera_infos if str(cam.get("name", "")) == name]

        if not found_devices:
            available_names = [f"{cam.get('name', 'Unknown')} (SN: {cam.get('id', 'Unknown')})" for cam in camera_infos]
            raise ValueError(
                f"No Gemini camera found with name '{name}'. Available cameras: {available_names}"
            )

        if len(found_devices) > 1:
            serial_numbers = [dev.get("id", "Unknown") for dev in found_devices]
            raise ValueError(
                f"Multiple Gemini cameras found with name '{name}'. "
                f"Please use a unique serial number instead. Found SNs: {serial_numbers}"
            )

        serial_number = str(found_devices[0].get("id", ""))
        return serial_number

    def _configure_pipeline_config(self) -> None:
        """Creates and configures the Orbbec pipeline configuration object."""
        if self.pipe_config is None:
            raise RuntimeError(f"{self}: pipe_config must be initialized before use.")

        if self.pipeline is None:
            raise RuntimeError(f"{self}: pipeline must be initialized before use.")

        # Enable color stream with specified or default parameters
        if self.width and self.height and self.fps:
            self.pipe_config.enable_video_stream(
                OBStreamType.COLOR_STREAM,
                self.capture_width,
                self.capture_height,
                self.fps,
                OBFormat.RGB
            )
        else:
            # Enable with default settings
            self.pipe_config.enable_video_stream(
                OBStreamType.COLOR_STREAM,
                0,  # 0 means use default
                0,
                0,
                OBFormat.RGB
            )

        # Enable depth stream if requested
        if self.use_depth:
            # Note: Depth stream may have different supported resolutions than color
            # We'll enable with default settings and let the camera decide
            self.pipe_config.enable_video_stream(
                OBStreamType.DEPTH_STREAM,
                0,  # Use default resolution
                0,
                0,
                OBFormat.Y16
            )

    def _configure_capture_settings(self) -> None:
        """Sets fps, width, and height from device stream if not already configured."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"Cannot validate settings for {self} as it is not connected.")

        if self.pipeline is None:
            raise RuntimeError(f"{self}: pipeline must be initialized before use.")

        # Get actual stream configuration by reading a frame
        try:
            frameset = self.pipeline.wait_for_frames(5000)
            if frameset:
                color_frame = frameset.get_color_frame()
                if color_frame:
                    actual_width = color_frame.get_width()
                    actual_height = color_frame.get_height()

                    # Always update to actual values from camera
                    if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                        self.width, self.height = actual_height, actual_width
                        self.capture_width, self.capture_height = actual_width, actual_height
                    else:
                        self.width, self.height = actual_width, actual_height
                        self.capture_width, self.capture_height = actual_width, actual_height

                    logger.info(f"{self} actual resolution: {self.width}x{self.height}")

                    # Get FPS if not set
                    if self.fps is None:
                        # Try to get from profile
                        try:
                            color_profiles = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
                            if color_profiles.get_count() > 0:
                                profile = color_profiles.get_default_video_stream_profile()
                                if profile:
                                    self.fps = profile.get_fps()
                        except:
                            self.fps = 30  # Default fallback
        except Exception as e:
            logger.warning(f"Could not get stream configuration for {self}: {e}. Using configured values.")

    def read_depth(self, timeout_ms: int = 2000) -> NDArray[Any]:
        """
        Reads a single depth frame synchronously from the camera.

        Args:
            timeout_ms (int): Maximum time in milliseconds to wait for a frame. Defaults to 2000ms.

        Returns:
            np.ndarray: The depth map as a NumPy array (height, width)
                  of type np.uint16 (raw depth values in millimeters) with rotation applied.

        Raises:
            DeviceNotConnectedError: If the camera is not connected.
            RuntimeError: If reading frames from the pipeline fails or depth is not enabled.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        if not self.use_depth:
            raise RuntimeError(
                f"Failed to capture depth frame '.read_depth()'. Depth stream is not enabled for {self}."
            )

        start_time = time.perf_counter()

        if self.pipeline is None:
            raise RuntimeError(f"{self}: pipeline must be initialized before use.")

        # Depth frames may not arrive every time, so we retry a few times
        max_attempts = 10
        for attempt in range(max_attempts):
            frameset = self.pipeline.wait_for_frames(timeout_ms)

            if frameset is None:
                continue

            depth_frame = frameset.get_depth_frame()
            if depth_frame is not None:
                # Get raw data and reshape to (height, width)
                # Depth data is in Y16 format (16-bit unsigned int)
                depth_data = np.asanyarray(depth_frame.get_data(), dtype=np.uint8)
                height = depth_frame.get_height()
                width = depth_frame.get_width()
                # Convert from byte array to uint16 array and reshape
                depth_data = depth_data.view(np.uint16).reshape((height, width))

                depth_map_processed = self._postprocess_image(depth_data, depth_frame=True)

                read_duration_ms = (time.perf_counter() - start_time) * 1e3
                logger.debug(f"{self} read_depth took: {read_duration_ms:.1f}ms (attempt {attempt+1})")

                return depth_map_processed

        raise RuntimeError(f"{self} read_depth failed: no depth frame received after {max_attempts} attempts.")

    def read_color_and_depth(self, color_mode: ColorMode | None = None, timeout_ms: int = 200) -> tuple[NDArray[Any], NDArray[Any] | None]:
        """
        Reads synchronized color and depth frames from the same frameset.

        Args:
            color_mode: Desired color mode for the output frame. If None, uses default.
            timeout_ms (int): Maximum time in milliseconds to wait for a frame. Defaults to 200ms.

        Returns:
            tuple: (color_image, depth_map) where depth_map is None if depth is not enabled.

        Raises:
            DeviceNotConnectedError: If the camera is not connected.
            RuntimeError: If reading frames from the pipeline fails.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start_time = time.perf_counter()

        if self.pipeline is None:
            raise RuntimeError(f"{self}: pipeline must be initialized before use.")

        frameset = self.pipeline.wait_for_frames(timeout_ms)

        if frameset is None:
            raise RuntimeError(f"{self} read failed: no frames received.")

        # Get color frame
        color_frame = frameset.get_color_frame()
        if color_frame is None:
            raise RuntimeError(f"{self} read failed: no color frame in frameset.")

        color_data = np.asanyarray(color_frame.get_data())
        height = color_frame.get_height()
        width = color_frame.get_width()
        color_data = color_data.reshape((height, width, 3))
        color_image_processed = self._postprocess_image(color_data, color_mode)

        # Get depth frame if enabled
        depth_map_processed = None
        if self.use_depth:
            depth_frame = frameset.get_depth_frame()
            if depth_frame is not None:
                depth_data = np.asanyarray(depth_frame.get_data(), dtype=np.uint8)
                d_height = depth_frame.get_height()
                d_width = depth_frame.get_width()
                depth_data = depth_data.view(np.uint16).reshape((d_height, d_width))
                depth_map_processed = self._postprocess_image(depth_data, depth_frame=True)

        read_duration_ms = (time.perf_counter() - start_time) * 1e3
        logger.debug(f"{self} read_color_and_depth took: {read_duration_ms:.1f}ms")

        return color_image_processed, depth_map_processed

    def read(self, color_mode: ColorMode | None = None, timeout_ms: int = 200) -> NDArray[Any]:
        """
        Reads a single color frame synchronously from the camera.

        Args:
            color_mode: Desired color mode for the output frame. If None, uses default.
            timeout_ms (int): Maximum time in milliseconds to wait for a frame. Defaults to 200ms.

        Returns:
            np.ndarray: The captured color frame as a NumPy array
              (height, width, channels), processed according to color_mode and rotation.

        Raises:
            DeviceNotConnectedError: If the camera is not connected.
            RuntimeError: If reading frames from the pipeline fails.
            ValueError: If an invalid color_mode is requested.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start_time = time.perf_counter()

        if self.pipeline is None:
            raise RuntimeError(f"{self}: pipeline must be initialized before use.")

        frameset = self.pipeline.wait_for_frames(timeout_ms)

        if frameset is None:
            raise RuntimeError(f"{self} read failed: no frames received.")

        color_frame = frameset.get_color_frame()
        if color_frame is None:
            raise RuntimeError(f"{self} read failed: no color frame in frameset.")

        # Get raw data and reshape to (height, width, channels)
        color_data = np.asanyarray(color_frame.get_data())
        height = color_frame.get_height()
        width = color_frame.get_width()
        color_data = color_data.reshape((height, width, 3))

        color_image_processed = self._postprocess_image(color_data, color_mode)

        read_duration_ms = (time.perf_counter() - start_time) * 1e3
        logger.debug(f"{self} read took: {read_duration_ms:.1f}ms")

        return color_image_processed

    def _postprocess_image(
        self, image: NDArray[Any], color_mode: ColorMode | None = None, depth_frame: bool = False
    ) -> NDArray[Any]:
        """
        Applies color conversion, dimension validation, and rotation to a raw frame.

        Args:
            image (np.ndarray): The raw image frame.
            color_mode (Optional[ColorMode]): The target color mode (RGB or BGR).
            depth_frame (bool): Whether this is a depth frame.

        Returns:
            np.ndarray: The processed image frame.

        Raises:
            ValueError: If the requested color_mode is invalid.
            RuntimeError: If the raw frame dimensions do not match configured dimensions.
        """
        if color_mode and color_mode not in (ColorMode.RGB, ColorMode.BGR):
            raise ValueError(
                f"Invalid requested color mode '{color_mode}'. Expected {ColorMode.RGB} or {ColorMode.BGR}."
            )

        if depth_frame:
            h, w = image.shape
            # Depth stream may have different resolution than color stream
            # Don't validate dimensions for depth frames
        else:
            h, w, c = image.shape
            if c != 3:
                raise RuntimeError(f"{self} frame channels={c} do not match expected 3 channels (RGB/BGR).")

            if h != self.capture_height or w != self.capture_width:
                raise RuntimeError(
                    f"{self} frame width={w} or height={h} do not match configured "
                    f"width={self.capture_width} or height={self.capture_height}."
                )

        processed_image = image

        # Orbbec SDK returns RGB format by default, convert to BGR if needed
        if not depth_frame and self.color_mode == ColorMode.BGR:
            processed_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]:
            processed_image = cv2.rotate(processed_image, self.rotation)

        return processed_image

    def _read_loop(self) -> None:
        """
        Internal loop run by the background thread for asynchronous reading.
        """
        if self.stop_event is None:
            raise RuntimeError(f"{self}: stop_event is not initialized before starting read loop.")

        while not self.stop_event.is_set():
            try:
                color_image = self.read(timeout_ms=500)

                with self.frame_lock:
                    self.latest_frame = color_image
                self.new_frame_event.set()

            except DeviceNotConnectedError:
                break
            except Exception as e:
                logger.warning(f"Error reading frame in background thread for {self}: {e}")

    def _start_read_thread(self) -> None:
        """Starts or restarts the background read thread if it's not running."""
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=0.1)
        if self.stop_event is not None:
            self.stop_event.set()

        self.stop_event = Event()
        self.thread = Thread(target=self._read_loop, args=(), name=f"{self}_read_loop")
        self.thread.daemon = True
        self.thread.start()

    def _stop_read_thread(self) -> None:
        """Signals the background read thread to stop and waits for it to join."""
        if self.stop_event is not None:
            self.stop_event.set()

        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)

        self.thread = None
        self.stop_event = None

    def async_read(self, timeout_ms: float = 200) -> NDArray[Any]:
        """
        Reads the latest available color frame asynchronously.

        Args:
            timeout_ms (float): Maximum time in milliseconds to wait for a frame.

        Returns:
            np.ndarray: The latest captured color frame.

        Raises:
            DeviceNotConnectedError: If the camera is not connected.
            TimeoutError: If no frame becomes available within the timeout.
            RuntimeError: If an unexpected error occurs.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self.thread is None or not self.thread.is_alive():
            self._start_read_thread()

        if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
            thread_alive = self.thread is not None and self.thread.is_alive()
            raise TimeoutError(
                f"Timed out waiting for frame from camera {self} after {timeout_ms} ms. "
                f"Read thread alive: {thread_alive}."
            )

        with self.frame_lock:
            frame = self.latest_frame
            self.new_frame_event.clear()

        if frame is None:
            raise RuntimeError(f"Internal error: Event set but no frame available for {self}.")

        return frame

    def disconnect(self) -> None:
        """
        Disconnects from the camera, stops the pipeline, and cleans up resources.

        Raises:
            DeviceNotConnectedError: If the camera is already disconnected.
        """
        if not self.is_connected and self.thread is None:
            raise DeviceNotConnectedError(
                f"Attempted to disconnect {self}, but it appears already disconnected."
            )

        if self.thread is not None:
            self._stop_read_thread()

        if self.pipeline is not None:
            self.pipeline.stop()
            self.pipeline = None
            self.pipe_config = None

        logger.info(f"{self} disconnected.")
