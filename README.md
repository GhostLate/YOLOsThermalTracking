# YOLOv5 | YOLOv7 object tracking (StrongSORT, ByteTrack, OCSORT)


## Introduction

This repository presents approach for tracking people using thermal imaging cameras.
For object detection was used YOLO family models (mostly YOLOv5 and YOLOv7 from [YOLOU](https://github.com/jizhishutong/YOLOU).
The detections generated by YOLO model are passed to one of three tracker:
[StrongSORT](https://github.com/dyhBUPT/StrongSORT)[](https://arxiv.org/abs/2202.13514),
[OCSORT](https://github.com/noahcao/OC_SORT)[](https://arxiv.org/abs/2203.14360) and 
[ByteTrack](https://github.com/ifzhang/ByteTrack)[](https://arxiv.org/abs/2110.06864). 
They can track any object that your YOLO model was trained to detect.

## Installation

```bash
sudo apt-get update && sudo apt upgrade
sudo apt install ffmpeg
pip install -r requirements.txt
git clone https://github.com/jizhishutong/YOLOU
cd ./YOLOU
pip install -r requirements.txt
```
I also suggest to use this [NVIDIA docker](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) container or 
just install manually **PyTorch**>=1.13, **CUDA**>=11.7, **TensorRT**>=8.5, **cuDNN**>=8.5

## Tracking

```bash
$ python3 inference.py --yolo-weights <weights path> --source <file(s) / folder path>
```

**Tracking methods:**

```bash
$ python3 inference.py --tracking-method ocsort
                                    bytetrack
                                    strongsort --reid-weights <ReID weights path>                
```

**Tracking sources:**

Tracking can be run on most video formats which [FFMPEG](https://ffmpeg.org/) supports.

```bash
$ python3 inference.py --source vid.mp4  # video
                           vid1.mp4, vid2.webm  # list of videos
                           path/  # directory
```

**YOLO models:**

```bash
$ python3 inference.py --yolo-weights yolov5m.pt
                                     yolov5m.engine
                                     yolov5s.pt
                                     yolov5s.engine
                                     yolov7s.pt (tiny version)
                                     yolov7s.engine (tiny version)
                                     yolov7.pt
                                     yolov7.engine
```

**ReID models**

For StrongSort tracking method you need to choose a ReID model based from this [ReID model zoo](https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO). 
These model can be further optimized for you needs by the `reid_export.py` script.

```bash
$ python3 track.py --source 0 --reid-weights osnet_x0_25_msmt17.pt
                                            osnet_x0_5_msmt17.engine
```
  
**Filter tracked classes:**

If you want to track only specific classes of objects, add their index after the `--classes` flag.

```bash
$ python3 track.py --source 0 --yolo-weights yolov5s.pt --classes 16 17  # COCO yolov5 model. Track cats and dogs, only
```

[Here](https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/) is a list of all the possible objects that a YOLO model trained on MS COCO can detect.


## Acknowledgements

<details><summary> <b>Expand</b> </summary>

* [https://github.com/mikel-brostrom/Yolov5_StrongSORT_OSNet](https://github.com/mikel-brostrom/Yolov5_StrongSORT_OSNet) (Trackers implementation)
* [https://github.com/jizhishutong/YOLOU](https://github.com/jizhishutong/YOLOU)  (YOLO models implementation)
* [https://github.com/WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)
* [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
* [https://github.com/ppogg/YOLOv5-Lite](https://github.com/ppogg/YOLOv5-Lite)
* [https://github.com/ceccocats/tkDNN](https://github.com/ceccocats/tkDNN)
* [https://github.com/pjreddie/darknet](https://github.com/pjreddie/darknet)
* [https://github.com/nebuly-ai/nebullvm](https://github.com/nebuly-ai/nebullvm)
* [https://github.com/luanshiyinyang/awesome-multiple-object-tracking](https://github.com/luanshiyinyang/awesome-multiple-object-tracking)
</details>