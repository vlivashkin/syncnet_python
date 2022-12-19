#!/usr/bin/python
import argparse
import glob
import os
import pickle
import shutil
import subprocess

from syncnet.config import Config
from syncnet.functions import face_detection, scene_detection, track_shot, crop_video
from syncnet.syncnet_instance import SyncNetInstance


class SyncNetMetric:
    def __init__(
        self,
        video_path: str,
        name: str = None,
        temp_dir="./temp",
        s3fd_weights_path="./weights/std_face.pth",
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
        command = "ffmpeg -hide_banner -y -i %s -qscale:v 2 -async 1 -r 25 %s" % (
            self.opt.videofile,
            os.path.join(self.opt.avi_dir, self.opt.reference, "video.avi"),
        )
        output = subprocess.call(command, shell=True, stdout=None)

        command = "ffmpeg -hide_banner -y -i %s -qscale:v 2 -threads 1 -f image2 %s" % (
            os.path.join(self.opt.avi_dir, self.opt.reference, "video.avi"),
            os.path.join(self.opt.frames_dir, self.opt.reference, "%06d.jpg"),
        )
        output = subprocess.call(command, shell=True, stdout=None)

        command = "ffmpeg -hide_banner -y -i %s -ac 1 -vn -acodec pcm_s16le -ar 16000 %s" % (
            os.path.join(self.opt.avi_dir, self.opt.reference, "video.avi"),
            os.path.join(self.opt.avi_dir, self.opt.reference, "audio.wav"),
        )
        output = subprocess.call(command, shell=True, stdout=None)

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
            vidtracks.append(
                crop_video(self.opt, track, os.path.join(self.opt.crop_dir, self.opt.reference, "%05d" % ii))
            )

        # ========== SAVE RESULTS ==========
        savepath = os.path.join(self.opt.work_dir, self.opt.reference, "tracks.pckl")
        with open(savepath, "wb") as fil:
            pickle.dump(vidtracks, fil)

        shutil.rmtree(os.path.join(self.opt.tmp_dir, self.opt.reference))

    def _inference(self):
        # ==================== LOAD MODEL AND FILE LIST ====================
        s = SyncNetInstance(device=self.device)
        s.load_parameters(self.opt.syncnet_weights_path)
        print("Model %s loaded." % self.opt.syncnet_weights_path)

        flist = glob.glob(os.path.join(self.opt.crop_dir, self.opt.reference, "0*.avi"))
        flist.sort()

        # ==================== GET OFFSETS ====================
        offsets, minvals, confs, dists = [], [], [], []
        for idx, fname in enumerate(flist):
            offset, minval, conf, dist = s.evaluate(self.opt, videofile=fname)
            offsets.append(offset)
            minvals.append(minval)
            confs.append(conf)
            dists.append(dist)

        # ==================== PRINT RESULTS TO FILE ====================
        with open(os.path.join(self.opt.work_dir, self.opt.reference, "activesd.pckl"), "wb") as fil:
            pickle.dump(dists, fil)

        return offsets, minvals, confs, dists

    def run(self):
        self._preprocessing_pipeline()
        offsets, minvals, confs, dists = self._inference()
        return offsets, minvals, confs, dists

    def cleanup(self):
        shutil.rmtree(self.opt.tmp_dir)


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
