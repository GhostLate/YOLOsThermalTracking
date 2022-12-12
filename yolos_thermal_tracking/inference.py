import argparse
import os
import sys
from pathlib import Path

import torch
import cv2
import numpy as np

# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["CUDA_MODULE_LOADING"] = "LAZY"

ROOT = Path(__file__).resolve().parents[0]  # YOLOU root directory

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'YOLOU') not in sys.path:
    sys.path.append(str(ROOT / 'YOLOU'))  # add YOLOU ROOT to PATH
if str(ROOT / 'trackers' / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'trackers' / 'strong_sort'))  # add strong_sort ROOT to PATH

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from YOLOU.utils.plots import Annotator, colors
from YOLOU.utils.general import scale_coords

from trackers.multi_tracker_zoo import create_tracker
from model import Detector
from dataloader import DataLoader
from visualiser import Visualiser
from custom_utils import time_sync, check_img_size
from media_utils import VideoSaver


@torch.no_grad()
def main(source, yolo_weights, tracking_method,
         device='', reid_weights=None, fp16=True, classes=None, imgsz=(640, 640), save_folder=None, show_results=True):
    # Load model
    detector = Detector(yolo_weights, half=fp16, device=device)

    # Create a Tracker instance
    tracker = create_tracker(tracking_method, reid_weights, detector.device, half=fp16)
    if hasattr(tracker, 'model'):
        if hasattr(tracker.model, 'warmup'):
            tracker.model.warmup()
    obj_tracks = {}

    imgsz = check_img_size(imgsz, s=detector.model.stride)  # check image size
    dataloader = DataLoader(source, img_size=imgsz, stride=detector.model.stride)

    video_saver = VideoSaver(save_folder)

    tracker_name = f"{tracking_method} - {Path(reid_weights).stem}" if reid_weights is not None else tracking_method
    visualiser = Visualiser(dataloader, detector.yolo_model_name, detector.class_names, tracker_name, obj_tracks)

    # Run inference
    curr_frame, prev_frame = None, None
    for mod_frame, orig_frame, meta_data in dataloader:
        # Get detections
        det, mod_frame, infer_dtime = detector(mod_frame, classes=classes)

        # Process predictions
        curr_frame = orig_frame

        annotator = Annotator(orig_frame, line_width=2, example=str(detector.class_names))

        # Camera Motion compensation
        t1 = time_sync()
        if hasattr(tracker, 'tracker.py') and hasattr(tracker.tracker, 'camera_update'):
            if prev_frame is not None and curr_frame is not None:
                tracker.tracker.camera_update(prev_frame, curr_frame)
        tracker_dtime = time_sync() - t1

        if det is not None and len(det):
            # Rescale boxes from mod_frame to orig_frame size
            det[:, :4] = scale_coords(mod_frame.shape[2:], det[:, :4], orig_frame.shape).round()

            # Pass detections to tracker.py
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
                        np.zeros(dataloader.video_files[dataloader.curr_file_id].nb_frames))
                    obj_tracks[dataloader.curr_file_id][(track_id, class_id)][dataloader.curr_frame_id] = conf
        else:
            t2 = time_sync()
            if hasattr(tracker, 'tracker.py') and hasattr(tracker.tracker, 'pred_n_update_all_tracks'):
                if prev_frame is not None and curr_frame is not None:
                    tracker.tracker.pred_n_update_all_tracks()
                    print(True)
            tracker_dtime += time_sync() - t2

        visualiser.draw(infer_dtime, det, tracker_dtime)

        orig_frame = annotator.result()

        if save_folder is not None:
            video_saver(orig_frame, meta_data)

        if show_results:
            cv2.imshow('tracking_results', orig_frame)
            cv2.waitKey(1)

        prev_frame = curr_frame
    visualiser.close()
    video_saver.close()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-yw', '--yolo-weights', type=str, default='./weights/yolov5m.engine',
        help='yolo model path (.pt / .engine)')
    parser.add_argument(
        '-s', '--source', nargs='+', default='./media/in',
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
        '--show-results', default=True,
        help='show tracking results on the screen')
    parser.add_argument(
        '--save-folder', type=str, default='./media/out',
        help='folder to save tracking results')
    parser.add_argument(
        '-tm', '--tracking-method', choices=['strongsort', 'ocsort', 'bytetrack'], default='strongsort',
        help='The tracker.py to choose (for strongsort require OSNet weights)')
    parser.add_argument(
        '-rw', '--reid-weights', type=str, default='./weights/osnet_x0_25_msmt17.engine',
        help='ReID model path (.pt / .engine). ONLY for strongsort tracking method')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    main(
        source=opt.source,
        yolo_weights=opt.yolo_weights,
        tracking_method=opt.tracking_method,
        reid_weights=opt.reid_weights,
        fp16=opt.fp16,
        device=opt.device,
        classes=opt.classes,
        imgsz=opt.imgsz,
        save_folder=opt.save_folder,
        show_results=opt.show_results
    )