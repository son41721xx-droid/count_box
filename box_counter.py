"""Count tracked boxes crossing an adjustable horizontal line in an RTSP stream."""

from __future__ import annotations

import argparse
import getpass
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, urlsplit, urlunsplit

# OpenCV reads these options when opening an FFmpeg RTSP stream.
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|stimeout;5000000"
)

import cv2
from ultralytics import YOLO


WINDOW_NAME = "YOLO Box Counter"


@dataclass
class TrackState:
    """Last stable side of the line occupied by one tracked object."""

    side: int
    last_seen: int


class AdjustableLine:
    def __init__(self, initial_ratio: float = 0.75) -> None:
        self.ratio = max(0.05, min(0.95, initial_ratio))
        self.y = 0
        self.frame_height = 0
        self.dragging = False
        self.changed = False

    def update_frame_height(self, height: int) -> None:
        if height != self.frame_height:
            self.frame_height = height
            self.y = int(height * self.ratio)

    def move_to(self, y: int) -> None:
        if self.frame_height <= 0:
            return
        new_y = max(0, min(self.frame_height - 1, y))
        if new_y != self.y:
            self.y = new_y
            self.ratio = self.y / self.frame_height
            self.changed = True

    def mouse_callback(self, event: int, _x: int, y: int, _flags: int, _param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and abs(y - self.y) <= 20:
            self.dragging = True
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.move_to(y)
        elif event == cv2.EVENT_LBUTTONUP:
            if self.dragging:
                self.move_to(y)
            self.dragging = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default=os.getenv(
            "BOX_COUNTER_RTSP",
            "rtsp://admin:Tce@sw1010@10.3.7.129:554/Streaming/Channels/102",
        ),
        help="RTSP URL (add username/password if the camera requires them)",
    )
    parser.add_argument(
        "--username",
        default=os.getenv("BOX_COUNTER_USERNAME"),
        help="Camera username (or set BOX_COUNTER_USERNAME)",
    )
    parser.add_argument(
        "--password",
        default=os.getenv("BOX_COUNTER_PASSWORD"),
        help="Camera password (prefer BOX_COUNTER_PASSWORD or the hidden prompt)",
    )
    parser.add_argument("--model", default="best.pt", help="Path to trained YOLO model")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.2, help="IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device, for example 0 for NVIDIA GPU or cpu",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Use FP16 inference (recommended with an NVIDIA GPU)",
    )
    parser.add_argument(
        "--line-ratio",
        type=float,
        default=0.75,
        help="Initial line position as fraction of image height (0-1)",
    )
    parser.add_argument(
        "--hysteresis",
        type=int,
        default=12,
        help="Pixels on each side of the line ignored to reduce jitter",
    )
    parser.add_argument(
        "--class-id",
        type=int,
        default=None,
        help="Only count this class ID; default counts every detected class",
    )
    parser.add_argument(
        "--tracker",
        default=str(Path(__file__).with_name("box_bytetrack.yaml")),
        help="Ultralytics tracker YAML (default keeps lost boxes longer)",
    )
    parser.add_argument(
        "--reconnect-after",
        type=int,
        default=10,
        help="Reconnect only after this many consecutive damaged/missing frames",
    )
    return parser.parse_args()


def authenticated_rtsp_url(source: str, username: str | None, password: str | None) -> str:
    """Insert URL-encoded credentials into an RTSP URL."""
    parsed = urlsplit(source)
    if parsed.scheme.lower() != "rtsp":
        raise ValueError("--source must begin with rtsp://")
    if parsed.username is not None or not username:
        return source

    if password is None:
        password = getpass.getpass("Camera password: ")

    credentials = f"{quote(username, safe='')}:{quote(password, safe='')}"
    return urlunsplit(parsed._replace(netloc=f"{credentials}@{parsed.netloc}"))


def open_stream(source: str) -> cv2.VideoCapture:
    # These timeouts prevent a dead camera from blocking the program forever.
    params = [
        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
        5000,
        cv2.CAP_PROP_READ_TIMEOUT_MSEC,
        5000,
    ]
    try:
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG, params)
    except (TypeError, cv2.error):
        # Compatibility fallback for older OpenCV builds.
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


class LatestFrameReader:
    """Continuously drain RTSP and expose only the newest decoded frame."""

    def __init__(self, source: str, reconnect_after: int) -> None:
        self.source = source
        self.reconnect_after = max(1, reconnect_after)
        self.condition = threading.Condition()
        self.frame = None
        self.sequence = 0
        self.generation = 0
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> "LatestFrameReader":
        self.thread.start()
        return self

    def read_latest(self, after_sequence: int, timeout: float = 2.0):
        """Wait for a frame newer than after_sequence."""
        deadline = time.monotonic() + timeout
        with self.condition:
            while self.sequence <= after_sequence and not self.stop_event.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None, after_sequence, self.generation
                self.condition.wait(remaining)
            return self.frame, self.sequence, self.generation

    def close(self) -> None:
        self.stop_event.set()
        with self.condition:
            self.condition.notify_all()
        self.thread.join(timeout=6)

    def _run(self) -> None:
        cap = open_stream(self.source)
        consecutive_failures = 0
        last_reconnect_message = 0.0

        try:
            while not self.stop_event.is_set():
                ok, frame = cap.read()
                if ok and frame is not None:
                    consecutive_failures = 0
                    with self.condition:
                        self.frame = frame
                        self.sequence += 1
                        self.condition.notify_all()
                    continue

                consecutive_failures += 1
                if consecutive_failures < self.reconnect_after:
                    self.stop_event.wait(0.01)
                    continue

                now = time.monotonic()
                if now - last_reconnect_message > 2:
                    print(
                        f"RTSP unavailable for {consecutive_failures} consecutive "
                        "frames; reconnecting..."
                    )
                    last_reconnect_message = now

                cap.release()
                if self.stop_event.wait(1):
                    break
                cap = open_stream(self.source)
                consecutive_failures = 0
                with self.condition:
                    self.generation += 1
                    self.condition.notify_all()
        finally:
            cap.release()


def stable_side(center_y: int, line_y: int, margin: int) -> int:
    """Return -1 above, +1 below, or 0 inside the dead band."""
    if center_y < line_y - margin:
        return -1
    if center_y > line_y + margin:
        return 1
    return 0


def main() -> None:
    args = parse_args()
    source = authenticated_rtsp_url(args.source, args.username, args.password)
    model_path = Path(args.model)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model not found: {model_path.resolve()}\n"
            "Put best.pt beside box_counter.py or pass --model C:\\path\\to\\best.pt"
        )

    model = YOLO(str(model_path))
    line = AdjustableLine(args.line_ratio)
    track_states: dict[int, TrackState] = {}
    down_count = 0
    up_count = 0
    frame_number = 0

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, line.mouse_callback)

    reader = LatestFrameReader(source, args.reconnect_after).start()
    last_sequence = 0
    last_generation = 0

    try:
        while True:
            frame, sequence, generation = reader.read_latest(last_sequence)
            if frame is None:
                continue

            # If inference is slower than the camera, skipped sequence numbers are
            # intentional: stale frames are discarded to keep latency near zero.
            last_sequence = sequence
            if generation != last_generation:
                # Do not compare new frames with object positions from before the outage.
                track_states.clear()
                last_generation = generation

            frame_number += 1
            height, width = frame.shape[:2]
            line.update_frame_height(height)

            # Moving the line must not create artificial crossings.
            if line.changed:
                track_states.clear()
                line.changed = False

            results = model.track(
                frame,
                persist=True,
                tracker=args.tracker,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                classes=None if args.class_id is None else [args.class_id],
                device=args.device,
                half=args.half,
                verbose=False,
            )

            result = results[0]
            boxes = result.boxes
            if boxes is not None and boxes.id is not None:
                coordinates = boxes.xyxy.int().cpu().tolist()
                track_ids = boxes.id.int().cpu().tolist()
                confidences = boxes.conf.cpu().tolist()
                class_ids = boxes.cls.int().cpu().tolist()

                for (x1, y1, x2, y2), track_id, confidence, class_id in zip(
                    coordinates, track_ids, confidences, class_ids
                ):
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    side = stable_side(center_y, line.y, args.hysteresis)
                    previous = track_states.get(track_id)

                    if side != 0:
                        if previous is not None and previous.side != side:
                            if previous.side == -1 and side == 1:
                                down_count += 1
                            elif previous.side == 1 and side == -1:
                                up_count += 1
                        track_states[track_id] = TrackState(side, frame_number)
                    elif previous is not None:
                        previous.last_seen = frame_number

                    class_name = result.names[class_id]
                    color = (0, 200, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.circle(frame, (center_x, center_y), 4, (0, 255, 255), -1)
                    cv2.putText(
                        frame,
                        f"{class_name} ID:{track_id} {confidence:.2f}",
                        (x1, max(25, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        color,
                        2,
                        cv2.LINE_AA,
                    )

            # Remove stale IDs so memory does not grow forever.
            track_states = {
                track_id: state
                for track_id, state in track_states.items()
                if frame_number - state.last_seen <= 300
            }

            net_count = down_count - up_count
            cv2.line(frame, (0, line.y), (width - 1, line.y), (0, 0, 255), 3)
            cv2.putText(
                frame,
                "Drag red line with mouse",
                (15, max(25, line.y - 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.rectangle(frame, (10, 10), (190, 120), (20, 20, 20), -1)
            cv2.putText(frame, f"DOWN  +1: {down_count}", (25, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            cv2.putText(frame, f"UP    -1: {up_count}", (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 255), 2)
            cv2.putText(frame, f"TOTAL    : {net_count}", (25, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            cv2.putText(frame, "Q: quit   R: reset", (10, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                down_count = 0
                up_count = 0
                track_states.clear()
    finally:
        reader.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
