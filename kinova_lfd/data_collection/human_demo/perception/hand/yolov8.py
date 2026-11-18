"""YOLOv8 hand detector (from WiLoR): the lightweight alternative to HaMeR's ViTDet + ViTPose.

Used inside the HaMeR server (``--detector yolov8``); it is not a server on its own. The weights
are WiLoR's ``detector.pt`` (``huggingface.co/spaces/rolpotamias/WiLoR``, ``pretrained_models/``),
saved as ``kinova_lfd/.cache/models/hamer/yolov8_hand_detector.pt``. It predicts hand boxes with a
left / right class directly, so it needs no person detector or keypoint model.

Written by Will Solow, 2026.
"""

from pathlib import Path

import numpy as np

from kinova_lfd.data_collection.human_demo.perception.hand.base import HandDetection, HandDetector
from kinova_lfd.data_collection.human_demo.perception.types import MODELS_DIR

CHECKPOINT_NAME = "yolov8_hand_detector.pt"
DEFAULT_CHECKPOINT = MODELS_DIR / "hamer" / CHECKPOINT_NAME


class Yolov8HandDetector(HandDetector):
    """Ultralytics YOLOv8 hand detector with left / right classes."""

    def __init__(
        self,
        models_dir: Path | None = None,
        device: str = "cuda",
        conf_thresh: float = 0.3,
        iou_thresh: float = 0.5,
    ) -> None:
        from ultralytics import YOLO

        checkpoint = DEFAULT_CHECKPOINT if models_dir is None else Path(models_dir) / CHECKPOINT_NAME
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"YOLOv8 hand detector {checkpoint} not found (WiLoR pretrained_models/detector.pt)."
            )
        self.model = YOLO(str(checkpoint))
        self.device = device
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        # WiLoR's detector uses class 0 = left, 1 = right; read the names in case a checkpoint differs.
        names = {int(i): str(n).lower() for i, n in self.model.names.items()}
        self.right_classes = {i for i, n in names.items() if "right" in n} or {1}

    def detect(self, rgb: np.ndarray) -> list[HandDetection]:
        bgr = np.ascontiguousarray(rgb[..., ::-1])  # ultralytics treats numpy input as BGR
        result = self.model.predict(bgr, conf=self.conf_thresh, iou=self.iou_thresh, device=self.device, verbose=False)[
            0
        ]
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        return [
            HandDetection(box=box.astype(np.float32), score=float(score), is_right=int(cls) in self.right_classes)
            for box, score, cls in zip(boxes, scores, classes)
        ]
