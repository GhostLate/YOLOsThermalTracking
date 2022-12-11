from pathlib import Path

import numpy as np
import torch
from YOLOU.models.common import DetectMultiBackend
from YOLOU.utils.general import non_max_suppression
from YOLOU.utils.torch_utils import select_device, TracedModel
from custom_utils import time_sync


class Detector:
    def __init__(self, yolo_weights: str | Path, trace=False, half: bool = False, device='', imgsz=(640, 640)):
        if isinstance(yolo_weights, Path):
            yolo_weights = str(Path(yolo_weights).resolve())
        self._device = select_device(device)
        self._model = DetectMultiBackend(yolo_weights, device=self._device, dnn=False, data=None, fp16=half)
        if trace:
            self._model = TracedModel(self._model, self._device, imgsz)
        self._class_names = self._model.names

        if hasattr(self._model, 'warmup'):
            bs = 1  # batch_size
            self._model.warmup(imgsz=(1 if self._model.pt else bs, 3, *imgsz))

        self._yolo_model_name = Path(self.model.weights).stem

    def __call__(self, im, aug=False, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic_nms=False, max_det=64):
        t1 = time_sync()
        im = torch.from_numpy(im).to(self._device)
        im = im.half() if self._model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        image_dtime = t2 - t1
        pred = self._model(im, aug)
        t3 = time_sync()
        model_dtime = t3 - t2
        if isinstance(pred, list):
            pred = pred[0]
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        t4 = time_sync()
        nms_dtime = t4 - t3
        return pred, im, np.array([image_dtime, model_dtime, nms_dtime])

    @property
    def device(self):
        return self._device

    @property
    def class_names(self):
        return self._class_names

    @property
    def model(self):
        return self._model

    @property
    def yolo_model_name(self):
        return self._yolo_model_name
