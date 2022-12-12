import os
from pathlib import Path

import cv2
import ffmpegio
import numpy as np

from media_utils import MetaData

VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv', 'webm'


class DataLoader:
    _curr_file_id: int
    _curr_frame_id: int
    _video_files: {int: MetaData}

    def __init__(self, source: str | Path | list, img_size=640, stride=32):
        self._video_files = {}
        file_id = 0
        for path in sorted(source) if isinstance(source, list) else [source]:
            if isinstance(source, Path):
                path = str(Path(path).resolve())
            if os.path.isdir(path):
                for root, dirs, files in os.walk(path, topdown=False):
                    for name in files:
                        if name.split('.')[-1].lower() in VID_FORMATS:
                            path = os.path.join(root, name)
                            meta_data = MetaData(path)
                            meta_data.file_path = path
                            self._video_files[file_id] = meta_data
                            file_id += 1
            elif os.path.isfile(path) and path.split('.')[-1].lower() in VID_FORMATS:
                meta_data = MetaData(path)
                meta_data.file_path = path
                self._video_files[file_id] = meta_data
                file_id += 1
        self.img_size = img_size
        self.stride = stride
        assert len(self._video_files) > 0, f'No videos found in {source}. Supported formats are:\n{VID_FORMATS}'

    def __iter__(self):
        for curr_file_id, meta_data in sorted(self._video_files.items()):
            self._curr_file_id = curr_file_id
            with ffmpegio.open(meta_data.file_path, 'rv', blocksize=100) as video_data:

                if not hasattr(meta_data, 'fps'):
                    if hasattr(video_data, 'rate'):
                        meta_data.fps = float(video_data.rate)
                    if hasattr(video_data, 'frame_rate'):
                        meta_data.fps = float(video_data.frame_rate)

                self._curr_frame_id = 0
                for frames_batch in video_data:
                    for frame in frames_batch:
                        mod_frame = letterbox(frame, self.img_size, stride=self.stride, auto=False)[0]  # padded resize
                        mod_frame = mod_frame.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
                        mod_frame = np.ascontiguousarray(mod_frame)  # contiguous
                        yield mod_frame, frame, meta_data
                        self._curr_frame_id += 1

    @property
    def curr_file_id(self):
        return self._curr_file_id

    @property
    def curr_frame_id(self):
        return self._curr_frame_id

    @property
    def video_files(self):
        return self._video_files


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)
