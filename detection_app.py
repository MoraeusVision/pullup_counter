import argparse
import cv2
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Optional

from inference import InferencePipeline
from inference.core.interfaces.camera.entities import VideoFrame
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmark, PoseLandmarksConnections
import supervision as sv

from utils import parse_video_source


logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

MODEL_PATH = "models/pose_landmarker_heavy.task"
VIDEO_PATH = "example_media/pullup_2.mp4"
OUTPUT_VIDEO_PATH = "output/saved_video.mp4"
MIN_VISIBILITY = 0.5


@dataclass(frozen=True)
class Keypoint2D:
    """
    Represents a single 2D keypoint (landmark) detected by MediaPipe.
    x, y: pixel coordinates
    visibility: confidence score (0-1)
    """
    x: int
    y: int
    visibility: float


@dataclass(frozen=True)
class KeypointsPrediction:
    """
    Holds all keypoints for a single frame, mapped by name.
    """
    keypoints: dict[str, Keypoint2D]


class KeypointsFrameRenderer:
    """
    Handles drawing keypoints and skeleton lines on video frames.
    """
    def __init__(self, min_visibility: float = MIN_VISIBILITY, draw_keypoint_names: bool = False):
        self.min_visibility = min_visibility
        self.draw_keypoint_names = draw_keypoint_names

    def render(self, prediction: Optional[KeypointsPrediction], video_frame: VideoFrame):
        """
        Draws keypoints and skeleton on the frame. Optionally draws keypoint names.
        """
        annotated_image = video_frame.image.copy()
        if prediction is None:
            return annotated_image

        # Draw skeleton lines between connected keypoints
        for connection in PoseLandmarksConnections.POSE_LANDMARKS:
            start_name = PoseLandmark(connection.start).name.lower()
            end_name = PoseLandmark(connection.end).name.lower()

            start = prediction.keypoints.get(start_name)
            end = prediction.keypoints.get(end_name)
            if start is None or end is None:
                continue

            if start.visibility < self.min_visibility or end.visibility < self.min_visibility:
                continue

            cv2.line(
                annotated_image,
                (start.x, start.y),
                (end.x, end.y),
                (0, 255, 0),
                2,
            )

        # Draw keypoints and optionally their names
        for name, keypoint in prediction.keypoints.items():
            if keypoint.visibility < self.min_visibility:
                continue

            cv2.circle(annotated_image, (keypoint.x, keypoint.y), 3, (0, 200, 255), -1)
            if self.draw_keypoint_names:
                cv2.putText(
                    annotated_image,
                    name,
                    (keypoint.x + 4, keypoint.y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        return annotated_image


class MediaPipePoseEstimator:
    """
    Wrapper around MediaPipe Tasks PoseLandmarker.
    Handles model initialization and keypoints inference on video/images.
    Returns keypoints in KeypointsPrediction format.
    """
    def __init__(
        self,
        min_pose_detection_confidence: float,
        min_pose_presence_confidence: float,
        min_tracking_confidence: float,
        model_path: str,
    ) -> None:
        # Check that the model file exists
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"MediaPipe Tasks model not found: {model_path}. "
                "Provide a valid .task model with --model-asset-path."
            )

        # Initialize MediaPipe Tasks PoseLandmarker
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=min_pose_detection_confidence,
            min_pose_presence_confidence=min_pose_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            num_poses=1,
            output_segmentation_masks=False,
        )
        self.pose = vision.PoseLandmarker.create_from_options(options)
        self._last_timestamp_ms = 0

    def _next_timestamp_ms(self) -> int:
        # Provides monotonically increasing timestamp for video inference
        now = int(time.monotonic() * 1000)
        self._last_timestamp_ms = max(self._last_timestamp_ms + 1, now)
        return self._last_timestamp_ms

    def predict(self, bgr_frame) -> Optional[KeypointsPrediction]:
        """
        Runs pose inference on an image and returns keypoints.
        """
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = self.pose.detect_for_video(mp_image, self._next_timestamp_ms())

        if not results.pose_landmarks:
            return None

        frame_height, frame_width = bgr_frame.shape[:2]
        keypoints: dict[str, Keypoint2D] = {}

        for index, landmark in enumerate(results.pose_landmarks[0]):
            name = PoseLandmark(index).name.lower()
            x = min(max(int(landmark.x * frame_width), 0), frame_width - 1)
            y = min(max(int(landmark.y * frame_height), 0), frame_height - 1)
            keypoints[name] = Keypoint2D(x=x, y=y, visibility=landmark.visibility)

        return KeypointsPrediction(keypoints=keypoints)

    def close(self) -> None:
        # Closes MediaPipe resources
        self.pose.close()


class FrameOutputManager:
    """
    Handles display (cv2.imshow), pause function, and video saving.
    Also handles keypresses (q for quit, space for pause).
    """
    def __init__(self, show: bool, save: bool):
        self.show = show
        self.save = save
        self.paused = False
        self.sink = None
        self.paused_frame = None

    def set_sink(self, sink) -> None:
        self.sink = sink

    def emit(self, frame, pipeline) -> None:
        """
        Displays and/or saves a frame depending on settings.
        """
        if self.show and frame is not None:
            self.visualize(frame, pipeline)

        if self.save and self.sink is not None and frame is not None:
            self.sink.write_frame(frame)

    def visualize(self, frame, pipeline) -> None:
        """
        Shows frame in a window. Space pauses, q stops the pipeline.
        """
        display_frame = self.paused_frame if self.paused and self.paused_frame is not None else frame
        cv2.imshow("Predictions", display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            pipeline.terminate()
            return

        if key == ord(" "):
            self.paused = not self.paused
            if self.paused:
                self.paused_frame = frame.copy()
            else:
                self.paused_frame = None
            logging.info("Paused" if self.paused else "Resumed")

class BaseKeypointsApp(ABC):
    """
    Base class for keypoints apps:
    - Connects video, estimator, renderer, and output manager.
    - Has a hook (process_predicted_frame) for project logic on keypoints.
    """
    def __init__(
        self,
        video_source: str,
        show: bool,
        save: bool,
        output_path: str,
        model_path: str,
        min_pose_detection_confidence: float,
        min_pose_presence_confidence: float,
        min_tracking_confidence: float,
        draw_keypoint_names: bool,
        smoothing_alpha: float = 0.75,
    ):
        self.show = show
        self.save = save
        self.output_path = output_path
        self.last_frame = None
        self.video_source = parse_video_source(video_source)

        self.alpha = smoothing_alpha
        self.prev_keypoints: Optional[dict[str, Keypoint2D]] = None

        # Initialize estimator (MediaPipe Tasks)
        self.estimator = MediaPipePoseEstimator(
            min_pose_detection_confidence=min_pose_detection_confidence,
            min_pose_presence_confidence=min_pose_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_path=model_path,
        )

        # Initialize renderer and output manager
        self.renderer = KeypointsFrameRenderer(
            min_visibility=MIN_VISIBILITY,
            draw_keypoint_names=draw_keypoint_names,
        )
        self.output_manager = FrameOutputManager(show=self.show, save=self.save)

        # Forbid saving from camera index (requires file source)
        if self.save and isinstance(self.video_source, int):
            raise ValueError("Saving camera streams requires a file source in this template.")

        if self.save and isinstance(self.video_source, str):
            self.video_info = sv.VideoInfo.from_video_path(self.video_source)

        # Initialize pipeline with custom hooks
        self.pipeline = InferencePipeline.init_with_custom_logic(
            video_reference=self.video_source,
            on_video_frame=self.infer,
            on_prediction=self.on_prediction,
        )

    @abstractmethod
    def process_predicted_frame(
        self,
        prediction: Optional[KeypointsPrediction],
        video_frame: VideoFrame,
    ) -> Optional[KeypointsPrediction]:
        """
        Hook for project logic on keypoints (e.g. pullup counter).
        """

    def on_prediction(self, prediction: Optional[KeypointsPrediction], video_frame: VideoFrame) -> None:
        """
        Runs for each frame after inference. Renders and handles output.
        """

        smoothed_prediction = self._apply_smoothing(prediction)

        processed_prediction = self.process_predicted_frame(prediction=smoothed_prediction, video_frame=video_frame)

        if self.show or self.save:
            rendered_frame = self.renderer.render(processed_prediction, video_frame)
            self.handle_rendered_outputs(rendered_frame)

    def infer(self, video_frames: list[VideoFrame]) -> list[Optional[KeypointsPrediction]]:
        """
        Runs inference on a batch of frames.
        """
        return [self.estimator.predict(video_frame.image) for video_frame in video_frames]

    def handle_rendered_outputs(self, frame) -> None:
        """
        Handles saving/displaying of rendered frame.
        """
        self.last_frame = frame
        self.output_manager.emit(frame, self.pipeline)

    def _apply_smoothing(
        self, prediction: Optional[KeypointsPrediction]
    ) -> Optional[KeypointsPrediction]:
        """Apply Exponential Moving Average (EMA) smoothing to reduce keypoint jitter."""
        if prediction is None:
            self.prev_keypoints = None
            return None

        if self.alpha >= 1.0:  # No smoothing
            return prediction

        if self.prev_keypoints is None:
            self.prev_keypoints = prediction.keypoints.copy()
            return prediction

        # Apply EMA smoothing
        smoothed: dict[str, Keypoint2D] = {}
        for name, kp in prediction.keypoints.items():
            prev = self.prev_keypoints.get(name)
            if prev is None:
                smoothed[name] = kp
                continue

            # Smooth x, y and visibility
            x = int(self.alpha * prev.x + (1 - self.alpha) * kp.x)
            y = int(self.alpha * prev.y + (1 - self.alpha) * kp.y)
            vis = self.alpha * prev.visibility + (1 - self.alpha) * kp.visibility

            smoothed[name] = Keypoint2D(x=x, y=y, visibility=vis)

        self.prev_keypoints = smoothed
        return KeypointsPrediction(keypoints=smoothed)

    def run(self) -> None:
        """
        Starts the pipeline and cleans up resources on exit.
        """
        try:
            if self.save:
                with sv.VideoSink(self.output_path, self.video_info, codec="H264") as sink:
                    self.output_manager.set_sink(sink)
                    self.pipeline.start()
                    self.pipeline.join()
            else:
                self.pipeline.start()
                self.pipeline.join()
        finally:
            self.estimator.close()
            cv2.destroyAllWindows()


class KeypointsApp(BaseKeypointsApp):
    """
    Simple default implementation of BaseKeypointsApp.
    Returns keypoints as-is without extra logic.
    """
    def process_predicted_frame(
        self,
        prediction: Optional[KeypointsPrediction],
        video_frame: VideoFrame,
    ) -> Optional[KeypointsPrediction]:
        return prediction


def parse_args():
    parser = argparse.ArgumentParser(description="MediaPipe keypoints template app")

    parser.add_argument("--source", type=str, default=VIDEO_PATH)
    parser.add_argument("--output", type=str, default=OUTPUT_VIDEO_PATH)

    parser.add_argument(
        "--model-path",
        type=str,
        default=MODEL_PATH,
        help="Path to MediaPipe Tasks Pose Landmarker .task model.",
    )
    parser.add_argument("--min-pose-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-pose-presence-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)

    parser.add_argument("--draw-keypoint-names", action="store_true", default=False)
    parser.add_argument("--show", action="store_true", default=False)
    parser.add_argument("--save", action="store_true", default=False)

    parser.add_argument(
        "--smoothing-alpha",
        type=float,
        default=0.1,
        help="Smoothing alpha (0-1) - lower number gives more smoothing",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    app = KeypointsApp(
        video_source=args.source,
        show=args.show,
        save=args.save,
        output_path=args.output,
        model_path=args.model_path,
        min_pose_detection_confidence=args.min_pose_detection_confidence,
        min_pose_presence_confidence=args.min_pose_presence_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
        draw_keypoint_names=args.draw_keypoint_names,
        smoothing_alpha=args.smoothing_alpha,
    )
    app.run()