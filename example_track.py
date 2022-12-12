from yolos_thermal_tracking.inference import main

if __name__ == "__main__":
    main(
        source='./media/in',
        yolo_weights='./weights/yolov5m.engine',
        tracking_method='strongsort',  # strongsort, ocsort, bytetrack
        reid_weights='./weights/osnet_x0_25_msmt17.engine',
        # osnet_x0_5_msmt17.engine, osnet_x0_5_msmt17.pt
        # osnet_x0_25_msmt17.engine, osnet_x0_25_msmt17.pt
        classes=[0],
        save_folder='../media/out'
    )

# YOLOv5: 'yolov5s.engine', 'yolov5s.pt', 'yolov5m.engine', 'yolov5m.pt'
# YOLOv7: 'yolov7s.engine', 'yolov7s.pt', 'yolov7.engine', 'yolov7.pt'

