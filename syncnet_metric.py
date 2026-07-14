import argparse
import json
import logging
import shutil
from typing import List

from syncnet.config import SyncNetConfig
from syncnet.ffmpeg import change_fps, extract_all_audio, extract_all_frames
from syncnet.functions import crop_video, face_detection, scene_detection, track_scene
from syncnet.instance import SyncNetInstance
from syncnet.s3fd.s3fd import S3FD

log = logging.getLogger(__name__)


class SyncNetMetric:
    def __init__(self, opt: SyncNetConfig):
        self.opt = opt

        self.s3fd = S3FD(weights_path=self.opt.s3fd_weights_path, device=self.opt.device)

        self.syncnet = SyncNetInstance(device=self.opt.device)
        self.syncnet.load_parameters(self.opt.syncnet_weights_path)

    def run(self, source_video_path, cleanup=True) -> List:
        video_path = f"{self.opt.data_dir}/resampled_video.mov"
        audio_path = f"{self.opt.data_dir}/resampled_audio.wav"
        change_fps(source_video_path, self.opt.frame_rate, video_path)
        extract_all_frames(video_path, f"{self.opt.frames_dir}/%06d.jpg")
        extract_all_audio(video_path, self.opt.audio_sample_rate, audio_path)

        faces = face_detection(self.opt.frames_dir, self.s3fd, self.opt)
        if self.opt.scene_detection:
            scenes = scene_detection(video_path)
        else:
            # Single-shot clip: one scene spanning all frames, no scenedetect pass.
            from types import SimpleNamespace

            scenes = [(SimpleNamespace(frame_num=0), SimpleNamespace(frame_num=len(faces)))]

        all_tracks = []
        for scene in scenes:
            if scene[1].frame_num - scene[0].frame_num >= self.opt.min_track:
                scene_tracks = track_scene(faces[scene[0].frame_num : scene[1].frame_num], self.opt)
                if self.opt.max_tracks_per_scene is not None and len(scene_tracks) > self.opt.max_tracks_per_scene:
                    continue  # Too many tracks in scene — skip ambiguous multi-person scenes
                all_tracks.extend(scene_tracks)

        if self.opt.single_track and len(all_tracks) > 1:
            # Keep only the dominant track: coverage (frames) x median face size — the subject,
            # not spurious micro-detections (background faces, figurines, posters).
            def _track_score(t):
                sizes = sorted(max(b[2] - b[0], b[3] - b[1]) for b in t["bbox"])
                return len(t["frame"]) * float(sizes[len(sizes) // 2])

            all_tracks = [max(all_tracks, key=_track_score)]

        tracks_scores = []
        for idx, track in enumerate(all_tracks):
            if len(track["frame"]) < 6:
                # evaluate() windows 5 frames (lastframe = min_length - 5): shorter tracks yield
                # zero windows and torch.cat crashes. Spurious micro-tracks carry no sync signal.
                continue
            cropped_video_path = f"{self.opt.crop_dir}/{idx:05d}.mov"
            crop_video(track, self.opt.frames_dir, audio_path, cropped_video_path, self.opt)
            offset, minval, conf = self.syncnet.evaluate(cropped_video_path, self.opt)
            tracks_scores.append(
                {
                    "timecodes": (track["frame"][0] / self.opt.frame_rate, track["frame"][-1] / self.opt.frame_rate),
                    "syncnet": {"offset": offset, "lse-d": minval, "lse-c": conf},
                }
            )

        if cleanup:
            shutil.rmtree(self.opt.data_dir)

        return tracks_scores


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s", level=logging.DEBUG)

    parser = argparse.ArgumentParser(description="SyncNet Metric")
    parser.add_argument("--video_path", type=str, required=True, help="")
    parser.add_argument("--temp_dir", type=str, default="./temp", help="")
    parser.add_argument("--output_path", type=str, help="")
    parser.add_argument("--device", type=str, default="cuda:0", help="")
    args = parser.parse_args()

    log.debug("Args:\n" + "\n".join(f"{k}: {v}" for k, v in args.__dict__.items()))

    opt = SyncNetConfig(data_dir=args.temp_dir, device=args.device)
    metric = SyncNetMetric(opt=opt)
    tracks_scores = metric.run(args.video_path)
    for track in tracks_scores:
        start, end = track["timecodes"]
        offset, lsed, lsec = track["syncnet"]["offset"], track["syncnet"]["lse-d"], track["syncnet"]["lse-c"]
        log.info(f"Scene [{start:.2f}:{end:.2f}] – AV offset: {offset}, LSE-D: {lsed:.3f}, LSE-C: {lsec:.3f}")

    if args.output_path is not None:
        with open(args.output_path, "w") as f:
            json.dump(tracks_scores, f, indent=4)
