#!/usr/bin/python
import argparse
import glob
import logging
import os
import pickle
import shutil
from typing import Tuple, List

import numpy as np

from syncnet.config import Config
from syncnet.functions import face_detection, scene_detection, track_shot, crop_video, call_ffmpeg
from syncnet.syncnet_instance import SyncNetInstance

log = logging.getLogger(__name__)


class SyncNetMetric:
    def __init__(
        self,
        video_path: str,
        name: str = None,
        temp_dir="./temp",
        s3fd_weights_path="./weights/sfd_face.pth",
        syncnet_weights_path="./weights/syncnet_v2.model",
        device="cuda:0",
    ):
        if name is None:
            name = video_path.rsplit("/", 1)[1].rsplit(".", 1)[0]
        self.opt = Config(
            video_path=video_path,
            name=name,
            temp_dir=temp_dir,
            s3fd_weights_path=s3fd_weights_path,
            syncnet_weights_path=syncnet_weights_path,
        )
        self.device = device

    def _preprocessing_pipeline(self):
        # ========== CONVERT VIDEO AND EXTRACT FRAMES ==========
        # fmt: off
        command = [
            "ffmpeg", "-hide_banner", "-y",
            "-i", self.opt.videofile,
            "-crf", "17",
            "-async", "1",
            "-r", f"{self.opt.frame_rate}",
            f"{self.opt.avi_dir}/{self.opt.reference}/video.mov",
        ]
        # fmt: on
        call_ffmpeg(command)

        # fmt: off
        command = [
            "ffmpeg", "-hide_banner", "-y",
            "-i", f"{self.opt.avi_dir}/{self.opt.reference}/video.mov",
            "-qscale:v", "2",
            "-threads", "1",
            "-f", "image2",
            f"{self.opt.frames_dir}/{self.opt.reference}/%06d.jpg",
        ]
        # fmt: on
        call_ffmpeg(command)

        # fmt: off
        command = [
            "ffmpeg", "-hide_banner", "-y",
            "-i", f"{self.opt.avi_dir}/{self.opt.reference}/video.mov",
            "-ac", "1",
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", f"{self.opt.audio_sample_rate}",
            f"{self.opt.avi_dir}/{self.opt.reference}/audio.wav"
        ]
        # fmt: on
        call_ffmpeg(command)

        # ========== FACE DETECTION ==========
        faces = face_detection(self.opt, self.device)

        # ========== SCENE DETECTION ==========
        scene = scene_detection(self.opt)

        # ========== FACE TRACKING ==========
        alltracks = []
        for shot in scene:
            if shot[1].frame_num - shot[0].frame_num >= self.opt.min_track:
                alltracks.extend(track_shot(self.opt, faces[shot[0].frame_num : shot[1].frame_num]))

        # ========== FACE TRACK CROP ==========
        vidtracks = []
        for ii, track in enumerate(alltracks):
            vidtracks.append(crop_video(self.opt, track, f"{self.opt.crop_dir}/{self.opt.reference}/{ii:05d}"))

        # ========== SAVE RESULTS ==========
        savepath = f"{self.opt.work_dir}/{self.opt.reference}/tracks.pckl"
        with open(savepath, "wb") as fil:
            pickle.dump(vidtracks, fil)

        shutil.rmtree(f"{self.opt.tmp_dir}/{self.opt.reference}")

    def _inference(self) -> Tuple[List[np.array], List[np.array], List[np.array], List[np.array]]:
        # ==================== LOAD MODEL AND FILE LIST ====================
        s = SyncNetInstance(device=self.device)
        s.load_parameters(self.opt.syncnet_weights_path)
        log.debug(f"Model {self.opt.syncnet_weights_path} loaded.")

        flist = glob.glob(os.path.join(self.opt.crop_dir, self.opt.reference, "0*.mov"))
        flist.sort()

        # ==================== GET OFFSETS ====================
        offsets, minvals, confs, dists = [], [], [], []
        for idx, fname in enumerate(flist):
            offset, minval, conf, dist = s.evaluate(self.opt, video_path=fname)
            offsets.append(offset)
            minvals.append(minval)
            confs.append(conf)
            dists.append(dist)

        # ==================== PRINT RESULTS TO FILE ====================
        with open(f"{self.opt.work_dir}/{self.opt.reference}/activesd.pckl", "wb") as fil:
            pickle.dump(dists, fil)

        return offsets, minvals, confs, dists

    def run(self) -> Tuple[List[np.array], List[np.array], List[np.array], List[np.array]]:
        self._preprocessing_pipeline()
        offsets, minvals, confs, dists = self._inference()
        return offsets, minvals, confs, dists

    def cleanup(self):
        shutil.rmtree(self.opt.data_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SyncNet Metric")
    parser.add_argument("--video_path", type=str, required=True, help="")
    parser.add_argument("--name", type=str, help="")
    parser.add_argument("--data_dir", type=str, default="./temp", help="")
    parser.add_argument("--device", type=str, default="cuda:0", help="")
    args = parser.parse_args()

    metric = SyncNetMetric(video_path=args.video_path, name=args.name, temp_dir=args.data_dir, device=args.device)
    offsets, minvals, confs, _ = metric.run()
    print(f"AV offset: {offsets}, LSE-D: {minvals}, LSE-C: {confs}")
