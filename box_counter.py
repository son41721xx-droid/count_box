"""Count tracked boxes entering or leaving an adjustable four-point area."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit, urlunsplit
from urllib.request import Request, urlopen

# OpenCV reads these options when opening an FFmpeg RTSP stream.
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|stimeout;5000000"
)

import cv2
import numpy as np
from ultralytics import YOLO


WINDOW_NAME = "YOLO Box Counter"
HEADER_HEIGHT = 130


@dataclass
class TrackState:
    """Last stable area membership of one tracked object."""

    inside: bool
    last_seen: int


class AdjustableArea:
    DEFAULT_RATIOS = (
        (0.20, 0.55),
        (0.80, 0.55),
        (0.80, 0.90),
        (0.20, 0.90),
    )

    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path
        self.ratios = list(self.DEFAULT_RATIOS)
        self.points: list[tuple[int, int]] = []
        self.frame_width = 0
        self.frame_height = 0
        self.dragging_index: int | None = None
        self.changed = False
        self.load()

    def load(self) -> None:
        """Load four normalized points from disk, falling back to defaults."""
        if not self.config_path.exists():
            return

        try:
            data = json.loads(self.config_path.read_text(encoding="utf-8"))
            ratios = data["ratios"]
            if not isinstance(ratios, list) or len(ratios) != 4:
                raise ValueError("ratios must contain exactly four points")

            validated: list[tuple[float, float]] = []
            for point in ratios:
                if not isinstance(point, list) or len(point) != 2:
                    raise ValueError("each point must contain x and y")
                x_ratio, y_ratio = float(point[0]), float(point[1])
                if not (0.0 <= x_ratio <= 1.0 and 0.0 <= y_ratio <= 1.0):
                    raise ValueError("point ratios must be between 0 and 1")
                validated.append((x_ratio, y_ratio))

            self.ratios = validated
            print(f"Loaded area from {self.config_path.resolve()}")
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            print(
                f"Warning: could not load area config {self.config_path.resolve()}: "
                f"{exc}. Using the default area."
            )

    def save(self) -> None:
        """Persist normalized points so the area survives program restarts."""
        data = {
            "version": 1,
            "ratios": [[x_ratio, y_ratio] for x_ratio, y_ratio in self.ratios],
        }
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = self.config_path.with_suffix(
                f"{self.config_path.suffix}.tmp"
            )
            temporary_path.write_text(
                json.dumps(data, indent=2), encoding="utf-8"
            )
            temporary_path.replace(self.config_path)
        except OSError as exc:
            print(
                f"Warning: could not save area config {self.config_path.resolve()}: "
                f"{exc}"
            )

    def update_frame_size(self, width: int, height: int) -> None:
        if width != self.frame_width or height != self.frame_height:
            self.frame_width = width
            self.frame_height = height
            self.points = [
                (int(width * x_ratio), int(height * y_ratio))
                for x_ratio, y_ratio in self.ratios
            ]

    def move_point(self, index: int, x: int, y: int) -> None:
        if self.frame_width <= 0 or self.frame_height <= 0:
            return
        x = max(0, min(self.frame_width - 1, x))
        y = max(0, min(self.frame_height - 1, y))
        if self.points[index] != (x, y):
            self.points[index] = (x, y)
            self.ratios[index] = (x / self.frame_width, y / self.frame_height)
            self.changed = True

    def reset(self) -> None:
        self.ratios = list(self.DEFAULT_RATIOS)
        if self.frame_width > 0 and self.frame_height > 0:
            self.points = [
                (int(self.frame_width * x_ratio), int(self.frame_height * y_ratio))
                for x_ratio, y_ratio in self.ratios
            ]
        self.changed = True

    @property
    def contour(self) -> np.ndarray:
        points = np.asarray(self.points, dtype=np.int32)
        center = points.mean(axis=0)
        angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
        points = points[np.argsort(angles)]
        return points.reshape((-1, 1, 2))

    def mouse_callback(self, event: int, x: int, y: int, _flags: int, param) -> None:
        # The displayed window has a status header above the camera image.
        # Convert window coordinates back to camera-frame coordinates.
        y_offset = int(param) if param is not None else 0
        y -= y_offset
        if event == cv2.EVENT_LBUTTONDOWN and self.points:
            if y < 0:
                return
            distances = [((px - x) ** 2 + (py - y) ** 2) ** 0.5 for px, py in self.points]
            nearest = int(np.argmin(distances))
            if distances[nearest] <= 25:
                self.dragging_index = nearest
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging_index is not None:
            self.move_point(self.dragging_index, x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            if self.dragging_index is not None:
                self.move_point(self.dragging_index, x, y)
                self.save()
            self.dragging_index = None


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
    parser.add_argument("--iou", type=float, default=0.3, help="IoU threshold")
    parser.add_argument("--imgsz", type=int, default=512, help="Inference image size")
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device, for example 0 for NVIDIA GPU or cpu",
    )
    parser.add_argument(
        "--quantize",
        type=int,
        choices=(16, 32),
        default=None,
        help="Inference precision: 16 for FP16 or 32 for FP32",
    )
    parser.add_argument("--half", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--hysteresis",
        type=int,
        default=12,
        help="Pixels around the area boundary ignored to reduce count jitter",
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
    parser.add_argument(
        "--area-config",
        default=str(Path(__file__).with_name("area_config.json")),
        help="File used to remember the last P1-P4 positions",
    )
    parser.add_argument(
        "--api-url",
        default="https://pwdapi.fec-corp.com/api/asrs/dock-front?dock=DOCK03",
        help="Dock API URL used to read order_no values",
    )
    parser.add_argument(
        "--api-token",
        default=os.getenv("DOCK_API_TOKEN"),
        help="Bearer token (prefer the DOCK_API_TOKEN environment variable)",
    )
    parser.add_argument(
        "--api-refresh",
        type=float,
        default=10.0,
        help="Seconds between Dock API requests",
    )
    parser.add_argument(
        "--box-count-url",
        default="https://pwdapi.fec-corp.com/api/asrs/dock-front/box-counts",
        help="POST endpoint used to send the current total box count",
    )
    parser.add_argument(
        "--post-retry",
        type=float,
        default=5.0,
        help="Seconds before retrying a failed box-count POST",
    )
    parser.add_argument(
        "--null-reset-after",
        type=float,
        default=10.0,
        help="Reset counts after order_no stays null for this many seconds",
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


def extract_order_numbers(payload: object) -> list[str]:
    """Return unique, non-empty order numbers from the API lanes."""
    if not isinstance(payload, dict):
        raise ValueError("API response must be a JSON object")
    lanes = payload.get("lanes")
    if not isinstance(lanes, list):
        raise ValueError("API response does not contain a lanes list")

    orders: list[str] = []
    seen: set[str] = set()
    for lane in lanes:
        if not isinstance(lane, dict):
            continue
        value = lane.get("order_no")
        if value is None:
            continue
        order_no = str(value).strip()
        if order_no and order_no not in seen:
            seen.add(order_no)
            orders.append(order_no)
    return orders


class DockOrderReader:
    """Refresh dock order numbers without blocking video processing."""

    def __init__(self, url: str, token: str, refresh_seconds: float) -> None:
        self.url = url
        self.token = token
        self.refresh_seconds = max(1.0, refresh_seconds)
        self.lock = threading.Lock()
        self.order_numbers: list[str] = []
        self.status = "CONNECTING"
        self.revision = 0
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> "DockOrderReader":
        self.thread.start()
        return self

    def snapshot(self) -> tuple[list[str], str, int]:
        with self.lock:
            return list(self.order_numbers), self.status, self.revision

    def close(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=3)

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                request = Request(
                    self.url,
                    headers={
                        "Accept": "application/json",
                        "Authorization": f"Bearer {self.token}",
                        "User-Agent": "YOLO-Box-Counter/1.0",
                    },
                )
                with urlopen(request, timeout=5) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                orders = extract_order_numbers(payload)
                with self.lock:
                    self.order_numbers = orders
                    self.status = "OK"
                    self.revision += 1
            except HTTPError as exc:
                with self.lock:
                    self.status = f"HTTP {exc.code}"
                print(f"Dock API request failed: HTTP {exc.code}")
            except (URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as exc:
                with self.lock:
                    self.status = "ERROR"
                print(f"Dock API request failed: {exc}")

            self.stop_event.wait(self.refresh_seconds)


class BoxCountSender:
    """POST the latest total in the background without delaying video."""

    def __init__(self, url: str, token: str, retry_seconds: float) -> None:
        self.url = url
        self.token = token
        self.retry_seconds = max(1.0, retry_seconds)
        self.condition = threading.Condition()
        self.generation = 0
        self.pending: tuple[int, str, int] | None = None
        self.status = "IDLE"
        self.stopping = False
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> "BoxCountSender":
        self.thread.start()
        return self

    def submit(self, order_no: str, total: int) -> None:
        # Keep only the newest total if counting changes faster than the API.
        with self.condition:
            self.pending = (self.generation, order_no, total)
            self.condition.notify_all()

    def cancel_pending(self) -> None:
        """Discard retries belonging to an order that is no longer active."""
        with self.condition:
            self.generation += 1
            self.pending = None
            if self.status != "SENDING":
                self.status = "IDLE"

    def snapshot(self) -> str:
        with self.condition:
            return self.status

    def close(self) -> None:
        with self.condition:
            self.stopping = True
            self.condition.notify_all()
        self.thread.join(timeout=6)

    def _run(self) -> None:
        while True:
            with self.condition:
                while self.pending is None and not self.stopping:
                    self.condition.wait()
                if self.pending is None and self.stopping:
                    return
                generation, order_no, total = self.pending
                self.pending = None
                self.status = "SENDING"

            body = json.dumps(
                {"order_no": order_no, "boxes": str(total)}
            ).encode("utf-8")
            try:
                request = Request(
                    self.url,
                    data=body,
                    method="POST",
                    headers={
                        "Accept": "application/json",
                        "Authorization": f"Bearer {self.token}",
                        "Content-Type": "application/json",
                        "User-Agent": "YOLO-Box-Counter/1.0",
                    },
                )
                with urlopen(request, timeout=5) as response:
                    response.read()
                with self.condition:
                    self.status = (
                        "OK" if generation == self.generation else "IDLE"
                    )
                print(f"Sent box count: order_no={order_no}, boxes={total}")
            except HTTPError as exc:
                with self.condition:
                    if generation == self.generation:
                        self.status = f"HTTP {exc.code}"
                        if self.pending is None:
                            self.pending = (generation, order_no, total)
                    else:
                        self.status = "IDLE"
                print(f"Box-count POST failed: HTTP {exc.code}")
            except (URLError, TimeoutError, OSError) as exc:
                with self.condition:
                    if generation == self.generation:
                        self.status = "ERROR"
                        if self.pending is None:
                            self.pending = (generation, order_no, total)
                    else:
                        self.status = "IDLE"
                print(f"Box-count POST failed: {exc}")

            if self.snapshot() != "OK":
                time.sleep(self.retry_seconds)


def stable_area_membership(
    point: tuple[int, int], contour: np.ndarray, margin: int
) -> bool | None:
    """Return True inside, False outside, or None inside the boundary dead band."""
    distance = cv2.pointPolygonTest(contour, point, True)
    if distance > margin:
        return True
    if distance < -margin:
        return False
    return None


def main() -> None:
    args = parse_args()
    # Backward-compatible CLI alias without passing deprecated `half` to Ultralytics.
    if args.half and args.quantize is None:
        args.quantize = 16
    source = authenticated_rtsp_url(args.source, args.username, args.password)
    model_path = Path(args.model)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model not found: {model_path.resolve()}\n"
            "Put best.pt beside box_counter.py or pass --model C:\\path\\to\\best.pt"
        )

    model = YOLO(str(model_path))
    area = AdjustableArea(Path(args.area_config))
    track_states: dict[int, TrackState] = {}
    enter_count = 0
    exit_count = 0
    frame_number = 0

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, area.mouse_callback, HEADER_HEIGHT)

    reader = LatestFrameReader(source, args.reconnect_after).start()
    order_reader: DockOrderReader | None = None
    count_sender: BoxCountSender | None = None
    if args.api_token:
        order_reader = DockOrderReader(
            args.api_url, args.api_token, args.api_refresh
        ).start()
        count_sender = BoxCountSender(
            args.box_count_url, args.api_token, args.post_retry
        ).start()
    else:
        print("Dock API disabled: set DOCK_API_TOKEN to read orders and send counts.")
    session_order_no: str | None = None
    last_submitted_total = 0
    last_order_revision = 0
    null_order_since: float | None = None
    null_order_reset_done = False
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
            area.update_frame_size(width, height)

            # Moving the area must not create artificial enter/exit events.
            if area.changed:
                track_states.clear()
                area.changed = False

            results = model.track(
                frame,
                persist=True,
                tracker=args.tracker,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                classes=None if args.class_id is None else [args.class_id],
                device=args.device,
                quantize=args.quantize,
                verbose=False,
            )

            result = results[0]
            boxes = result.boxes

            # Draw the editable area before boxes so live detections stay clear.
            overlay = frame.copy()
            cv2.fillPoly(overlay, [area.contour], (0, 0, 255))
            cv2.addWeighted(overlay, 0.14, frame, 0.86, 0, frame)
            cv2.polylines(frame, [area.contour], True, (0, 0, 255), 3)
            for index, (point_x, point_y) in enumerate(area.points, start=1):
                cv2.circle(frame, (point_x, point_y), 9, (0, 255, 255), -1)
                cv2.circle(frame, (point_x, point_y), 11, (0, 0, 255), 2)
                cv2.putText(
                    frame,
                    f"P{index}",
                    (point_x + 12, point_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

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
                    inside = stable_area_membership(
                        (center_x, center_y), area.contour, max(0, args.hysteresis)
                    )
                    previous = track_states.get(track_id)

                    if inside is not None:
                        if previous is not None and previous.inside != inside:
                            if inside:
                                enter_count += 1
                            else:
                                exit_count += 1
                        track_states[track_id] = TrackState(inside, frame_number)
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

            net_count = enter_count - exit_count
            if order_reader is None:
                order_numbers, api_status, order_revision = [], "NO TOKEN", 0
            else:
                order_numbers, api_status, order_revision = order_reader.snapshot()

            # Only successful API responses can start or continue the null timer.
            # Requiring new revisions avoids treating one stale empty response as
            # continuous confirmation that the order is still null.
            if (
                api_status not in ("OK", "CONNECTING")
                and not null_order_reset_done
            ):
                # An API failure breaks the continuous-null observation.
                null_order_since = None
            if api_status == "OK" and order_revision != last_order_revision:
                last_order_revision = order_revision
                if order_numbers:
                    null_order_since = None
                    null_order_reset_done = False
                    if session_order_no is None:
                        session_order_no = order_numbers[0]
                        print(
                            "Using initial order_no for box counts: "
                            f"{session_order_no}"
                        )
                else:
                    now = time.monotonic()
                    if null_order_since is None:
                        null_order_since = now
                    elif (
                        not null_order_reset_done
                        and now - null_order_since >= max(0.0, args.null_reset_after)
                    ):
                        enter_count = 0
                        exit_count = 0
                        net_count = 0
                        track_states.clear()
                        session_order_no = None
                        last_submitted_total = 0
                        if count_sender is not None:
                            count_sender.cancel_pending()
                        null_order_reset_done = True
                        print(
                            "order_no stayed null for "
                            f"{args.null_reset_after:g} seconds; total count reset."
                        )
            if (
                count_sender is not None
                and session_order_no is not None
                and net_count != last_submitted_total
            ):
                count_sender.submit(session_order_no, net_count)
                last_submitted_total = net_count
            post_status = count_sender.snapshot() if count_sender else "NO TOKEN"
            cv2.putText(
                frame,
                "Drag yellow points P1-P4 to adjust area",
                (15, height - 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            header = np.full((HEADER_HEIGHT, width, 3), (20, 20, 20), dtype=np.uint8)
            cv2.putText(header, f"ENTER +1: {enter_count}", (25, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            cv2.putText(header, f"EXIT  -1: {exit_count}", (25, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 255), 2)
            cv2.putText(header, f"TOTAL    : {net_count}", (25, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

            if order_numbers:
                display_orders = order_numbers
            elif null_order_reset_done:
                display_orders = ["NULL - TOTAL RESET"]
            elif null_order_since is not None:
                null_seconds = time.monotonic() - null_order_since
                display_orders = [
                    f"NULL {min(null_seconds, args.null_reset_after):.0f}/"
                    f"{args.null_reset_after:g}s"
                ]
            else:
                display_orders = ["-"]
            status_color = (0, 255, 0) if api_status == "OK" else (0, 200, 255)
            cv2.putText(header, f"ORDER NO  API:{api_status} POST:{post_status}", (265, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.52, status_color, 2, cv2.LINE_AA)
            for order_index, order_no in enumerate(display_orders[:4]):
                cv2.putText(header, order_no, (265, 55 + order_index * 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "Q: quit   R: reset count   A: reset area", (10, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            display_frame = np.vstack((header, frame))
            cv2.imshow(WINDOW_NAME, display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                enter_count = 0
                exit_count = 0
                track_states.clear()
            if key == ord("a"):
                area.reset()
                area.save()
                track_states.clear()
    finally:
        if count_sender is not None:
            count_sender.close()
        if order_reader is not None:
            order_reader.close()
        reader.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
