"""ONNX inference wrapper for the YOLO26 end-to-end cat detector.
The exported YOLO26 head already does decoding + NMS inside the graph, so the
runtime only has to letterbox the input, run the session, and map boxes back
to original-image pixel coordinates.
"""
from __future__ import annotations
from pathlib import Path
from typing import Sequence
import cv2
import numpy as np
import onnxruntime as ort
class CatDetector:
    def __init__(
        self,
        onnx_path: str | Path,
        imgsz: int = 640,
        conf: float = 0.25,
        class_names: Sequence[str] = ("cat",),
    ) -> None:
        self.onnx_path = str(onnx_path)
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.class_names = tuple(class_names)
        self.session = ort.InferenceSession(
            self.onnx_path, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.output_shape = tuple(self.session.get_outputs()[0].shape)
    def _letterbox(self, img_bgr: np.ndarray):
        """Resize-with-pad to (imgsz, imgsz) using Ultralytics' exact LetterBox
        formula and return (padded_rgb, scale, left_pad, top_pad) so we can
        invert the transform on the output boxes."""
        h0, w0 = img_bgr.shape[:2]
        r = min(self.imgsz / h0, self.imgsz / w0)
        new_w, new_h = int(round(w0 * r)), int(round(h0 * r))
        dw, dh = (self.imgsz - new_w) / 2, (self.imgsz - new_h) / 2
        # Ultralytics asymmetric half-pixel rounding: total padding always sums
        # to an integer, avoiding accumulation of off-by-one in the inverse.
        top    = int(round(dh - 0.1));  bottom = int(round(dh + 0.1))
        left   = int(round(dw - 0.1));  right  = int(round(dw + 0.1))
        resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        padded  = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=(114, 114, 114),
        )
        rgb = padded[:, :, ::-1]          # BGR → RGB
        return rgb, r, left, top
    def predict(self, image_path: str | Path) -> list[dict]:
        img_bgr = cv2.imread(str(image_path))
        h0, w0 = img_bgr.shape[:2]
        rgb, scale, left, top = self._letterbox(img_bgr)
        x = rgb.astype(np.float32) / 255.0
        x = x.transpose(2, 0, 1)[None, ...]        # HWC → BCHW
        raw = self.session.run([self.output_name], {self.input_name: x})[0]
        if raw.ndim == 3:
            detections = raw[0]
        else:
            detections = raw
        results: list[dict] = []
        for det in detections:
            if det.shape[0] < 6:
                continue
            x1, y1, x2, y2, score, cls = (
                float(det[0]), float(det[1]), float(det[2]), float(det[3]),
                float(det[4]), int(det[5]),
            )
            if score < self.conf:
                continue
            x1 = (x1 - left) / scale
            y1 = (y1 - top)  / scale
            x2 = (x2 - left) / scale
            y2 = (y2 - top)  / scale
            x1 = max(0.0, min(float(w0), x1))
            y1 = max(0.0, min(float(h0), y1))
            x2 = max(0.0, min(float(w0), x2))
            y2 = max(0.0, min(float(h0), y2))
            if x2 <= x1 or y2 <= y1:
                continue
            cls_name = (
                self.class_names[cls]
                if 0 <= cls < len(self.class_names)
                else str(cls)
            )
            results.append(
                {
                    "xmin": x1,
                    "ymin": y1,
                    "xmax": x2,
                    "ymax": y2,
                    "confidence": score,
                    "class": cls_name,
                }
            )
        return results
