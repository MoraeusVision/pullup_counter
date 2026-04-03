import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
from inference.core.interfaces.camera.entities import VideoFrame

from detection_app import (
    BaseKeypointsApp,
    Keypoint2D,
    KeypointsPrediction,
    MIN_VISIBILITY,
)


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

DEFAULT_MODEL_PATH = "models/pose_landmarker_heavy.task"
DEFAULT_SOURCE = "example_media/pullup_2.mp4"
DEFAULT_OUTPUT = "output/pullup_counter.mp4"


@dataclass(frozen=True)
class PullUpFrameSignals:
    """Aggregated frame signals used by the counter state machine."""

    face_y: int
    hands_y: int


class PullUpCounterStateMachine:
    """Counts pull-ups when face keypoints pass above hand keypoints."""

    def __init__(self, hysteresis_px: int) -> None:
        self.count: int = 0
        self.hysteresis_px = max(0, hysteresis_px)
        self._was_face_above_hands: bool = False

    def update(self, signals: PullUpFrameSignals) -> bool:
        """
        Update state from current frame signals.

        Returns True only on an upward crossing event, which represents
        one completed pull-up according to the requested rule.
        """
        upper_threshold = signals.hands_y - self.hysteresis_px
        lower_threshold = signals.hands_y + self.hysteresis_px

        has_crossed_up = signals.face_y < upper_threshold
        has_crossed_down = signals.face_y > lower_threshold

        if not self._was_face_above_hands and has_crossed_up:
            self.count += 1
            self._was_face_above_hands = True
            return True

        if self._was_face_above_hands and has_crossed_down:
            self._was_face_above_hands = False

        return False


class PullUpCounterOverlay:
    """Renders pull-up counter text only after the first registered pull-up."""

    def __init__(self) -> None:
        self._font = cv2.FONT_HERSHEY_SIMPLEX
        self._origin = (24, 64)
        self._font_scale = 2.0
        self._thickness = 4

    def draw(self, frame, count: int) -> None:
        if count < 1:
            return

        # Draw shadow for readability across bright backgrounds.
        label = f"Pull-ups: {count}"
        cv2.putText(
            frame,
            label,
            self._origin,
            self._font,
            self._font_scale,
            (0, 0, 0),
            self._thickness + 2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            label,
            self._origin,
            self._font,
            self._font_scale,
            (255, 255, 255),
            self._thickness,
            cv2.LINE_AA,
        )


class PullUpCounterApp(BaseKeypointsApp):
    """Application that counts pull-ups using MediaPipe pose keypoints."""

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
        smoothing_alpha: float,
        min_landmark_visibility: float,
        hysteresis_px: int,
    ) -> None:
        super().__init__(
            video_source=video_source,
            show=show,
            save=save,
            output_path=output_path,
            model_path=model_path,
            min_pose_detection_confidence=min_pose_detection_confidence,
            min_pose_presence_confidence=min_pose_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            draw_keypoint_names=draw_keypoint_names,
            smoothing_alpha=smoothing_alpha,
        )
        self.min_landmark_visibility = min_landmark_visibility
        self.counter = PullUpCounterStateMachine(hysteresis_px=hysteresis_px)
        self.overlay = PullUpCounterOverlay()

    def _get_keypoint_if_visible(
        self,
        prediction: KeypointsPrediction,
        name: str,
    ) -> Optional[Keypoint2D]:
        keypoint = prediction.keypoints.get(name)
        if keypoint is None or keypoint.visibility < self.min_landmark_visibility:
            return None
        return keypoint

    def _extract_signals(
        self,
        prediction: KeypointsPrediction,
    ) -> Optional[PullUpFrameSignals]:
        """Extracts the face/hand vertical positions for counter logic."""
        nose = self._get_keypoint_if_visible(prediction, "nose")
        left_wrist = self._get_keypoint_if_visible(prediction, "left_wrist")
        right_wrist = self._get_keypoint_if_visible(prediction, "right_wrist")

        if nose is None or left_wrist is None or right_wrist is None:
            return None

        hands_y = int((left_wrist.y + right_wrist.y) / 2)
        return PullUpFrameSignals(face_y=nose.y, hands_y=hands_y)

    def process_predicted_frame(
        self,
        prediction: Optional[KeypointsPrediction],
        video_frame: VideoFrame,
    ) -> Optional[KeypointsPrediction]:
        if prediction is None:
            return None

        signals = self._extract_signals(prediction)
        if signals is not None:
            did_increment = self.counter.update(signals)
            if did_increment:
                logging.info("Pull-up count: %d", self.counter.count)

        self.overlay.draw(video_frame.image, self.counter.count)
        return prediction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pull-up counter app built from keypoint template")

    parser.add_argument("--source", type=str, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)

    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_MODEL_PATH,
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
    parser.add_argument(
        "--min-landmark-visibility",
        type=float,
        default=MIN_VISIBILITY,
        help="Minimum keypoint visibility used by pull-up logic.",
    )
    parser.add_argument(
        "--hysteresis-px",
        type=int,
        default=12,
        help="Pixel margin around hand height used to stabilize up/down crossings.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not Path(args.model_path).exists():
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    app = PullUpCounterApp(
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
        min_landmark_visibility=args.min_landmark_visibility,
        hysteresis_px=args.hysteresis_px,
    )
    app.run()


if __name__ == "__main__":
    main()