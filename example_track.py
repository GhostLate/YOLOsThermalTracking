import argparse

from yolos_thermal_tracking.inference import main


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-yw', '--yolo-weights', type=str, default='./weights/yolov5s.engine',
        help='yolo model path (.pt / .engine)')
    parser.add_argument(
        '-s', '--source', nargs='+', default='./media/',
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
        '--save-folder', type=str, default=None,
        help='folder to save tracking results')
    parser.add_argument(
        '-tm', '--tracking-method', choices=['strongsort', 'ocsort', 'bytetrack'], default='ocsort',
        help='The tracker.py to choose (for strongsort require OSNet weights)')
    parser.add_argument(
        '-rw', '--reid-weights', type=str, default='./weights/osnet_x0_25_msmt17.pt',
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
        classes=opt.classes,
        imgsz=opt.imgsz,
        save_folder=opt.save_folder,
        show_results=opt.show_results
    )

# YOLOv5: 'yolov5s.engine', 'yolov5s.pt', 'yolov5m.engine', 'yolov5m.pt'
# YOLOv7: 'yolov7s.engine', 'yolov7s.pt', 'yolov7.engine', 'yolov7.pt'
# OSNet: 'osnet_x0_5_msmt17.engine', 'osnet_x0_5_msmt17.pt', 'osnet_x0_25_msmt17.engine', 'osnet_x0_25_msmt17.pt'