from fractions import Fraction
import time
import av, cv2, numpy as np
import imageio.v3 as iio
from typing import Callable, List, Tuple, Optional

from .utils import _load_default_plate_model, _predict_plates

Box = Tuple[int, int, int, int, float] 

def _estimate_fps(cap) -> float:
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if 1.0 <= fps <= 240.0:
        return fps
    # Sample first ~2 seconds to estimate fps
    start_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    frames = 0
    t0 = cap.get(cv2.CAP_PROP_POS_MSEC)  # ms at current pos
    while frames < 120:  # up to ~4s at 30fps
        ok, _ = cap.read()
        if not ok: break
        frames += 1
        t1 = cap.get(cv2.CAP_PROP_POS_MSEC)
        if (t1 - t0) >= 1500:  # ≥1.5 sec observed
            break
    # reset to original position
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_pos)
    if frames >= 2 and (t1 - t0) > 1:
        return max(1e-3, 1000.0 * frames / (t1 - t0))
    return 25.0  # last resort


def _fps_fraction(fps: float) -> Fraction:
    # preserve odd rates precisely; snap common NTSC
    for val, frac in [(23.976, Fraction(24000,1001)),
                      (29.97,  Fraction(30000,1001)),
                      (59.94,  Fraction(60000,1001))]:
        if abs(fps - val) < 0.02:
            return frac
    return Fraction(fps).limit_denominator(1001)



def _even_wh(w: int, h: int): return w - (w % 2), h - (h % 2)


def _draw_boxes(
    frame_bgr: np.ndarray,
    boxes: List[Box],
    label: str = "plate",
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """
    Draws bounding boxes and confidence on the frame (in-place).
    """
    for (x1, y1, x2, y2, score) in boxes:
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, thickness)
        text = f"{label} {score:.2f}"
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y_text = max(y1, th + 5)
        cv2.rectangle(frame_bgr, (x1, y_text - th - 4), (x1 + tw + 4, y_text + baseline), color, -1)
        cv2.putText(frame_bgr, text, (x1 + 2, y_text - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, lineType=cv2.LINE_AA)
    return frame_bgr


class AvWriter:
    def __init__(self, path: str, fps: float, width: int, height: int):
        fr = _fps_fraction(fps)
        self.w, self.h = _even_wh(width, height)
        self.container = av.open(path, mode="w")
        self.stream = self.container.add_stream("libx264", rate=fr)
        self.stream.width  = self.w
        self.stream.height = self.h
        self.stream.pix_fmt = "yuv420p"
        # self.stream.time_base = Fraction(1, fr)  # usually set from rate; explicit not required

    def write_bgr(self, bgr: np.ndarray):
        if (bgr.shape[1], bgr.shape[0]) != (self.w, self.h):
            bgr = cv2.resize(bgr, (self.w, self.h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frame = av.VideoFrame.from_ndarray(rgb, format="rgb24").reformat(
            width=self.w, height=self.h, format="yuv420p")
        for pkt in self.stream.encode(frame):
            self.container.mux(pkt)

    def close(self):
        for pkt in self.stream.encode():  # flush
            self.container.mux(pkt)
        self.container.close()
        

def _ensure_even_size(img):
    H, W = img.shape[:2]
    W2 = W - (W % 2)
    H2 = H - (H % 2)
    if (W2, H2) != (W, H):
        img = cv2.resize(img, (W2, H2), interpolation=cv2.INTER_AREA)
    return img


def video_read_write(
    video_path: str,
    model_version: str,
    configs: tuple,
    output_path: Optional[str] = None,
    flip_horizontal: bool = True,
    show_progress: bool = True,
) -> Optional[str]:

    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print("Error opening video file")
        return None
    
    data_config, training_config = configs
    width  = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = _estimate_fps(video)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    if not np.isfinite(fps) or fps <= 0:
        fps = 25.0

    if output_path is None:
        base, _ = os.path.splitext(video_path)
        output_path = f"{base}_out.mp4"

    # Prepare predictor
    model, encoder, transform, training_config = _load_default_plate_model(model_version, data_config, training_config)
    predict_fn = lambda frm: _predict_plates(model, frm, encoder, transform, training_config)

    writer = AvWriter(output_path, fps, width, height)

    t0 = time.time()
    frame_idx = 0

    try:
        while True:
            ok, frame = video.read()
            if not ok:
                break

            out_frame = np.ascontiguousarray(frame[:, ::-1, :] if flip_horizontal else frame)

            # predict on a COPY to avoid in-place mutations
            raw = predict_fn(out_frame.copy())
            boxes = []
            for b in raw:
                bb = np.asarray(b, dtype=np.float64).reshape(-1)
                if bb.size < 4 or np.any(~np.isfinite(bb[:4])):
                    continue
                x1, y1, x2, y2 = bb[:4]
                score = float(bb[4]) if bb.size > 4 else 1.0
                boxes.append((int(np.rint(x1)), int(np.rint(y1)),
                              int(np.rint(x2)), int(np.rint(y2)), score))

            _draw_boxes(out_frame, boxes, label="plate", color=(0, 255, 0), thickness=2)

            # Sanity: even dims, uint8, RGB for imageio
            out_frame = _ensure_even_size(out_frame)
            if out_frame.dtype != np.uint8:
                out_frame = np.clip(out_frame, 0, 255).astype(np.uint8, copy=False)
            rgb = cv2.cvtColor(out_frame, cv2.COLOR_BGR2RGB)

            # w.write(rgb)  # imageio expects RGB frames
            writer.write_bgr(out_frame)

            frame_idx += 1
            if show_progress and frame_idx % 50 == 0:
                if total_frames > 0:
                    print(f"Processed {frame_idx}/{total_frames} frames ({100.0*frame_idx/total_frames:.1f}%)")
                else:
                    print(f"Processed {frame_idx} frames...")

    finally:
        video.release()
        try:
            writer.close()
        except Exception as e:
            print("close writer warning:", e)

    if show_progress:
        dt = time.time() - t0
        print(f"Done. Wrote: {output_path}  |  Frames: {frame_idx}  |  Time: {dt:.2f}s  |  FPS(out): {fps:.2f}")

    return output_path
