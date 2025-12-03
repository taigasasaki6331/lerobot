## 変更ファイル一覧

1. GeminiCameraConfig の登録
    ```
    src/lerobot/cameras/gemini/configration_gemini.py
    @CameraConfig.register_subclass("gemini")  # 追加
    @dataclass(kw_only=True)
    class GeminiCameraConfig(CameraConfig):
    ```
2. lerobot_teleoperateでGeminiCameraConfigをインポート
    ```
    src/lerobot/scripts/lerobot_teleoperate.py
    from lerobot.cameras.gemini.configration_gemini import GeminiCameraConfig  # noqa: F401  # 追加
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
    from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
    ```
3. GeminiCameraに同期読み込みメソッドを追加
    ```
    src/lerobot/cameras/gemini/camera_gemini.py
    def read_color_and_depth(self, color_mode: ColorMode | None = None, timeout_ms: int = 200) -> tuple[NDArray[Any], NDArray[Any] | None]:
        """
        Reads synchronized color and depth frames from the same frameset.
        """
        # 1つのframesetからcolorとdepthの両方を取得
        frameset = self.pipeline.wait_for_frames(timeout_ms)
        
        # colorフレーム取得
        color_frame = frameset.get_color_frame()
        # ... 処理 ...
        
        # depthフレーム取得（同じframeset）
        if self.use_depth:
            depth_frame = frameset.get_depth_frame()
            # ... 処理 ...
        
        return color_image_processed, depth_map_processed
    ```
4. SO101Followerの修正
    
    4a. observation_featuresにdepthを追加
    ```
    src/lerobot/robots/so101_follower/so101_follower.py
    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        features = {}
        for cam_key, cam_config in self.config.cameras.items():
            # Add color image
            features[cam_key] = (cam_config.height, cam_config.width, 3)
            # Add depth image if enabled
            if hasattr(cam_config, 'use_depth') and cam_config.use_depth:
                features[f"{cam_key}.depth"] = (cam_config.height, cam_config.width)  # 追加
        return features
    ```
    4b. フレームキャッシュの追加
    ```
    src/lerobot/robots/so101_follower/so101_follower.py
    self.cameras = make_cameras_from_configs(config.cameras)
    # Cache for last successful camera frames
    self._last_camera_frames: dict[str, Any] = {}  # 追加
    ```
    4c. get_observationでdepth読み込み + キャッシュ機能
    ```
    src/lerobot/robots/so101_follower/so101_follower.py
    # For cameras with depth, use synchronized read to avoid frameset conflicts
    if hasattr(cam_config, 'use_depth') and cam_config.use_depth and hasattr(cam, 'read_color_and_depth'):
        try:
            # Read color and depth from the same frameset
            color_img, depth_img = cam.read_color_and_depth()  # 同期読み込み
            
            obs_dict[cam_key] = color_img
            if depth_img is not None:
                obs_dict[f"{cam_key}.depth"] = depth_img
            
            # Cache successful frames
            self._last_camera_frames[cam_key] = color_img
            if depth_img is not None:
                self._last_camera_frames[f"{cam_key}.depth"] = depth_img
        except Exception as e:
            logger.warning(f"Failed to read from {cam_key}: {e}")
            # Use cached frames if available
            if cam_key in self._last_camera_frames:
                obs_dict[cam_key] = self._last_camera_frames[cam_key]
                if f"{cam_key}.depth" in self._last_camera_frames:
                    obs_dict[f"{cam_key}.depth"] = self._last_camera_frames[f"{cam_key}.depth"]
    ```
    4d. PIDゲインの変更
    ```
    src/lerobot/robots/so101_follower/so101_follower.py
    # Set P_Coefficient to lower value to avoid shakiness (Default is 32)
    self.bus.write("P_Coefficient", motor, 8)  # default 16 → 8に変更
    # Set I_Coefficient and D_Coefficient to default value 0 and 32
    self.bus.write("I_Coefficient", motor, 8)  # default 0 → 8に追加
    self.bus.write("D_Coefficient", motor, 16)  # default 32 → 16に変更
    ```

5. テレオペ実行用シェルスクリプトの作成
    ```
    executables/gemini_teleoperate.sh
    --robot.cameras="{ front: {type: gemini, serial_number_or_name: 'CPBG152000D1', width: 1280, height: 720, fps: 30, use_depth: true}}"
    ```

## 主な変更ポイント
Geminiカメラの登録: @CameraConfig.register_subclass("gemini")で認識可能に
同期読み込み: read_color_and_depth()で1つのframesetからcolor+depthを取得し、ちらつきを解消
フレームキャッシュ: 読み込み失敗時に前回のフレームを使用
Depthサポート: observation_featuresに{cam}.depthを追加
PIDゲイン調整: P=8, I=8, D=16に変更
