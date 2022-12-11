import json
import os.path
import subprocess
from pathlib import Path

import cv2
import numpy as np


def ffprobe(filename, **kwargs):
    args = ['ffprobe', '-hide_banner', '-show_format', '-show_streams', '-of', 'json']
    for k, v in kwargs.items():
        args += [f'-{k}', f'{v}']
    args += [filename]

    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    if p.returncode != 0:
        raise ('ffprobe', out, err)
    return json.loads(out.decode('utf-8'))


class MetaData:
    nb_frames: int
    width: int
    height: int
    duration_ms: float
    fps: float
    file_path: str

    def __init__(self, source: str, show_entries=False, force_get_nb_frames=True):
        self.source = source

        if show_entries:
            probe = ffprobe(source, select_streams='v', show_entries='frame=coded_picture_number,pkt_pts_time')
        else:
            probe = ffprobe(source)
        self.format = probe['format']
        self.video_stream = [stream for stream in probe['streams'] if stream['codec_type'] == 'video'][0]

        self.file_name = probe['format']['filename']
        if 'nb_frames' in self.video_stream:
            self.nb_frames = int(self.video_stream['nb_frames'])
        elif force_get_nb_frames:
            probe = ffprobe(source, select_streams='v', show_entries='frame=coded_picture_number')
            self.nb_frames = len(probe['frames'])

        self.width = int(self.video_stream["width"])
        self.height = int(self.video_stream["height"])
        if show_entries and "duration" in self.video_stream:
            self.duration_ms = float(self.video_stream["duration"]) * 1000
            self.fps = float((self.nb_frames + 1) / self.duration_ms * 1000)
            self.timestamps = np.zeros(len(probe['frames']), dtype=np.int)
            for frame in probe['frames']:
                if 'pkt_pts_time' in frame:
                    self.timestamps[frame['coded_picture_number']] = int(float(frame['pkt_pts_time']) * 1000)


class VideoSaver:
    _curr_video_path: str
    __video_writer: cv2.VideoWriter

    def __init__(self, save_folder: str | Path):
        if isinstance(save_folder, Path):
            save_folder = str(Path(save_folder).resolve())
        self.save_folder = save_folder
        if save_folder is not None:
            os.makedirs(save_folder, exist_ok=True)

    def __call__(self, frame: np.ndarray, curr_meta_data: MetaData):
        if not hasattr(self, '_curr_video_path'):
            self.__update_video_writer(frame, curr_meta_data)

        if self._curr_video_path != curr_meta_data.file_path:
            self.__video_writer.release()
            self.__update_video_writer(frame, curr_meta_data)
        self.__video_writer.write(frame)

    def __update_video_writer(self, frame, curr_meta_data: MetaData):
        self._curr_video_path = curr_meta_data.file_path

        save_path = self.get_save_path(curr_meta_data.file_path)

        self.__video_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), curr_meta_data.fps,
                                              (frame.shape[1], frame.shape[0]))

    def get_save_path(self, file_path: str):
        save_path = os.path.join(self.save_folder, Path(file_path).stem + '_new.mp4')
        while os.path.isfile(save_path):
            save_path = self.get_save_path(save_path)
        return save_path

    def close(self):
        if self.save_folder is not None:
            self.__video_writer.release()
