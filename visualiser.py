import sys

import numpy as np
import plotly.graph_objects as go

from dataloader import DataLoader


class Visualiser:
    __curr_file_id: int
    __curr_frame_id: int
    __print_rows: int
    __curr_det: np.ndarray | None

    def __init__(self, dataloader: DataLoader, yolo_model_name: str, class_names: list, tracker_name, obj_tracks: dict):
        self.class_names = class_names
        self.yolo_model_name = yolo_model_name
        self.tracker_name = tracker_name

        self.dataloader = dataloader
        self.obj_tracks = obj_tracks

        self.video_files = dataloader.video_files

        self._total_dtime = {}

    def __update_data(self, infer_dtime: np.ndarray, det: np.ndarray = None, tracker_dtime: float = None):
        if not hasattr(self, '_Visualiser__curr_file_id'):
            self.__curr_file_id = self.dataloader.curr_file_id
        else:
            if self.__curr_file_id != self.dataloader.curr_file_id:
                self.__draw_heatmap()
            else:
                self.__clear(self.__print_rows)
        self.__print_rows = 8
        self.__curr_file_id = self.dataloader.curr_file_id
        self.__curr_frame_id = self.dataloader.curr_frame_id

        self._total_dtime.setdefault(self.__curr_file_id, {})
        self._total_dtime[self.__curr_file_id].setdefault(self.yolo_model_name, {'dtime': 0, 'ntime': 0})
        self._total_dtime[self.__curr_file_id][self.yolo_model_name]['dtime'] += infer_dtime
        self._total_dtime[self.__curr_file_id][self.yolo_model_name]['ntime'] += 1

        if tracker_dtime is not None:
            self._total_dtime[self.__curr_file_id].setdefault(self.tracker_name, {'dtime': 0, 'ntime': 0})
            self._total_dtime[self.__curr_file_id][self.tracker_name]['dtime'] += tracker_dtime
            self._total_dtime[self.__curr_file_id][self.tracker_name]['ntime'] += 1

        self.__curr_det = det

    def draw(self, infer_dtime: np.ndarray, det: np.ndarray = None, tracker_dtime: float = None):
        self.__update_data(infer_dtime, det, tracker_dtime)

        # Print Header
        info = f'\n Video: {self.__curr_file_id + 1}/{len(self.video_files)}' \
               f' Frame: {self.__curr_frame_id + 1}/{self.video_files[self.__curr_file_id][1].nb_frames}' \
               f' File Path: {self.video_files[self.__curr_file_id][0]}'
        print(info)

        # Print detections per class
        info = f' Detections:'
        if det is not None and len(det):
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()
                info += f" {n} {self.class_names[int(c)]}{'s' * (n > 1)},"
            info += f' Done.'
        else:
            info += f' None.'
        print(info)

        # Print inference dTime
        info = f'\n {f"Current Speed:":<17}'
        infer_dtime *= 1000
        info += f'{f"{self.yolo_model_name} (img/model/nms): ({infer_dtime[0]:.2f}/{infer_dtime[1]:.2f}/{infer_dtime[2]:.2f}(ms))":<50}'
        if tracker_dtime is not None:
            tracker_dtime *= 1000
            info += f'{f"{self.tracker_name}: ({tracker_dtime:.2f}ms)":<20}'
        else:
            info += f'{f"{self.tracker_name}: (None)":<20}'

        info += f'\n {f"Mean Speed:":<17}'
        dtime = self._total_dtime[self.__curr_file_id][self.yolo_model_name]
        total_time = dtime['dtime'] / dtime['ntime'] * 1000
        info += f'{f"{self.yolo_model_name} (img/model/nms): ({total_time[0]:.2f}/{total_time[1]:.2f}/{total_time[2]:.2f}(ms))":<50}'
        dtime = self._total_dtime[self.__curr_file_id][self.tracker_name]
        total_time = dtime['dtime'] / dtime['ntime'] * 1000
        info += f'{f"{self.tracker_name}: ({total_time:.2f}ms)":<20}'
        print(info)

        # Print table of tracks
        info = f'\n TrackID | ClassID : TrackFrames | TotalFrames'
        if self.__curr_file_id in self.obj_tracks:
            for (track_id, class_id), obj_track in self.obj_tracks[self.__curr_file_id].items():
                info += f'\n {f"{track_id}":<7} | {f"{class_id}":<7}'
                info += f' : {f"{len(np.where(obj_track > 0)[0])}":<11} | {f"{self.__curr_frame_id}":<11}'
                self.__print_rows += 1
        print(info)

    def __clear(self, rows):
        sys.stdout.write(f'\033[{rows}A')
        sys.stdout.write(f'\033[J')

    def __print_line(self, y, text):
        sys.stdout.write(f'\033[{self.__print_rows}A')
        sys.stdout.write(f'\033[K')
        print(text)

    def close(self):
        self.__draw_heatmap()

    def __draw_heatmap(self):
        np_data = np.zeros((len(self.obj_tracks[self.__curr_file_id].keys()), self.__curr_frame_id), dtype=float)

        titels = []
        for obj_track_id, track_key in enumerate(self.obj_tracks[self.__curr_file_id].keys()):
            np_data[obj_track_id] = self.obj_tracks[self.__curr_file_id][track_key][:self.__curr_frame_id]
            titels.append(str({track_key}))

        fig = go.Figure(data=go.Heatmap(
            z=np_data,
            y=titels,
            x=[*range(np_data.shape[1])],
            colorscale='Viridis'))

        fig.update_layout(
            template="plotly_dark",
            xaxis_title="Frames",
            yaxis_title="Track ID | Class ID",
            title={
                'text':
                    f"Trackers Viewer: "
                    f"{self.dataloader.video_files[self.__curr_file_id][0]} "
                    f"({self.yolo_model_name} | {self.tracker_name})",
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 20}
            }
        )
        fig.show()

    @property
    def total_dtime(self):
        return self._total_dtime
