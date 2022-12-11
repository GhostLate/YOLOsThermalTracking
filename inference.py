import argparse
import os

import cv2
import numpy as np

# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["CUDA_MODULE_LOADING"] = "LAZY"

import sys
from pathlib import Path
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOU root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'YOLOU') not in sys.path:
    sys.path.append(str(ROOT / 'YOLOU'))  # add YOLOU ROOT to PATH
if str(ROOT / 'trackers' / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'trackers' / 'strong_sort'))  # add strong_sort ROOT to PATH
if str(ROOT / 'trackers' / 'smile_track') not in sys.path:
    sys.path.append(str(ROOT / 'trackers' / 'smile_track'))  # add strong_sort ROOT to PATH

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from YOLOU.utils.plots import Annotator, colors
from YOLOU.utils.general import scale_coords

from trackers.multi_tracker_zoo import create_tracker
from model import Detector
from dataloader import DataLoader
from visualiser import Visualiser
from custom_utils import time_sync, check_img_size


@torch.no_grad()
def main(source, yolo_weights, tracking_method, reid_weights=None, fp16=True, classes=None, imgsz=(640, 640)):
    # Load model
    detector = Detector(yolo_weights, half=fp16)
    imgsz = check_img_size(imgsz, s=detector.model.stride)  # check image size

    # Initialize a DataLoader
    dataloader = DataLoader(source, img_size=imgsz, stride=detector.model.stride)

    # Create a Tracker instance
    tracker = create_tracker(tracking_method, reid_weights, detector.device, half=fp16)
    if hasattr(tracker, 'model'):
        if hasattr(tracker.model, 'warmup'):
            tracker.model.warmup()
    obj_tracks = {}

    # Create a Visualizer
    visualiser = Visualiser(dataloader, detector.yolo_model_name, detector.class_names, tracking_method, obj_tracks)

    # Run inference
    curr_frame, prev_frame = None, None
    for mod_frame, orig_frame in dataloader:
        # Inference
        pred, mod_frame, infer_dtime = detector(mod_frame, classes=classes)

        # Process predictions
        for det in pred:  # per image in the batch
            curr_frame = orig_frame

            annotator = Annotator(orig_frame, line_width=2, example=str(detector.class_names))

            # Camera Motion compensation
            t1 = time_sync()
            if hasattr(tracker, 'tracker') and hasattr(tracker.tracker, 'camera_update'):
                if prev_frame is not None and curr_frame is not None:
                    tracker.tracker.camera_update(prev_frame, curr_frame)
            tracker_dtime = time_sync() - t1

            if det is not None and len(det):
                # Rescale boxes from mod_frame to orig_frame size
                det[:, :4] = scale_coords(mod_frame.shape[2:], det[:, :4], orig_frame.shape).round()

                # pass detections to tracker
                t2 = time_sync()
                outputs = tracker.update(det.cpu(), orig_frame)  # [x1, y1, x2, y2, track_id, class_id, conf, queue]
                tracker_dtime += time_sync() - t2

                # Draw results and save the detected tracks
                if len(outputs) > 0:
                    for j, (output, conf) in enumerate(zip(outputs, det[:, 4])):
                        bbox, class_id, track_id, conf = output[0:4], int(output[5]), int(output[4]), output[6]

                        label = f'{track_id}, {detector.class_names[class_id]}, {conf:.2f}'
                        annotator.box_label(bbox, label, color=colors(class_id, True))

                        obj_tracks.setdefault(dataloader.curr_file_id, {})
                        obj_tracks[dataloader.curr_file_id].setdefault(
                            (track_id, class_id),
                            np.zeros(dataloader.video_files[dataloader.curr_file_id][1].nb_frames))
                        obj_tracks[dataloader.curr_file_id][(track_id, class_id)][dataloader.curr_frame_id] = conf
            else:
                t2 = time_sync()
                if hasattr(tracker, 'tracker') and hasattr(tracker.tracker, 'pred_n_update_all_tracks'):
                    if prev_frame is not None and curr_frame is not None:
                        tracker.tracker.pred_n_update_all_tracks()
                tracker_dtime += time_sync() - t2

            visualiser.draw(infer_dtime, det, tracker_dtime)

            orig_frame = annotator.result()
            cv2.imshow('test', orig_frame)
            cv2.waitKey(1)

            prev_frame = curr_frame
    visualiser.close()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-yw', '--yolo-weights', type=str, default=ROOT / 'weights/yolov5s.engine',
        help='yolo model path (.pt / .engine)')
    parser.add_argument(
        '-s', '--source', nargs='+', default=ROOT / 'test.webm',
        help='file/files or folder')
    parser.add_argument(
        '--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640, 640],
        help='inference size h,w')
    parser.add_argument(
        '--device', default='',
        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument(
        '-c', '--classes', nargs='+', type=int, default=[0],
        help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument(
        '--fp16', default=True,
        help='use FP16 half-precision inference')
    parser.add_argument(
        '-tm', '--tracking-method', choices=['strongsort', 'ocsort', 'bytetrack'], default='ocsort',
        help='The tracker to choose (for strongsort require OSNet weights)')
    parser.add_argument(
        '-rw', '--reid-weights', type=str, default=ROOT / 'weights/osnet_x0_25_msmt17.pt',
        help='ReID model path (.pt / .engine). ONLY for strongsort tracking method')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    main(opt.source, opt.yolo_weights, opt.tracking_method, opt.reid_weights, opt.fp16, opt.classes, opt.imgsz)


# YOLOv5: 'yolov5s.engine', 'yolov5s.pt', 'yolov5m.engine', 'yolov5m.pt'
# YOLOv7: 'yolov7s.engine', 'yolov7s.pt', 'yolov7.engine', 'yolov7.pt'
# OSNet: 'osnet_x0_5_msmt17.engine', 'osnet_x0_5_msmt17.pt', 'osnet_x0_25_msmt17.engine', 'osnet_x0_25_msmt17.pt'