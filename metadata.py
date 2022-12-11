import json
import subprocess

import numpy as np


def ffprobe(filename, **kwargs):
    args = ['ffprobe',  '-hide_banner', '-show_format', '-show_streams', '-of', 'json']
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
            self.fpms = (self.nb_frames + 1) / self.duration_ms
            self.timestamps = np.zeros(len(probe['frames']), dtype=np.int)
            for frame in probe['frames']:
                if 'pkt_pts_time' in frame:
                    self.timestamps[frame['coded_picture_number']] = int(float(frame['pkt_pts_time']) * 1000)
