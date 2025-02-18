import glob
import logging
import os
from typing import Dict, List, Tuple

import cv2
import numpy as np
from scenedetect import FrameTimecode
from scenedetect.detectors import ContentDetector
from scenedetect.scene_manager import SceneManager
from scenedetect.stats_manager import StatsManager
from scenedetect.video_manager import VideoManager
from scipy import signal
from scipy.interpolate import interp1d
from tqdm.auto import tqdm

from syncnet.config import SyncNetConfig
from syncnet.ffmpeg import combine_video_and_audio, crop_audio
from syncnet.s3fd.s3fd import S3FD

log = logging.getLogger(__name__)


def bb_intersection_over_union(boxA: np.array, boxB: np.array) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou


def track_scene(scenefaces: np.array, opt: SyncNetConfig) -> List[Dict]:
    iouThres = 0.5  # Minimum IOU between consecutive face detections
    tracks = []

    while True:
        track = []
        for framefaces in scenefaces:
            for face in framefaces:
                if len(track) == 0:
                    track.append(face)
                    framefaces.remove(face)
                elif face["frame"] - track[-1]["frame"] <= opt.num_failed_det:
                    iou = bb_intersection_over_union(face["bbox"], track[-1]["bbox"])
                    if iou > iouThres:
                        track.append(face)
                        framefaces.remove(face)
                        continue
                else:
                    break

        if len(track) == 0:
            break
        elif len(track) > opt.min_track:
            framenum = np.array([f["frame"] for f in track])
            bboxes = np.array([np.array(f["bbox"]) for f in track])

            frame_i = np.arange(framenum[0], framenum[-1] + 1)

            bboxes_i = []
            for ij in range(0, 4):
                interpfn = interp1d(framenum, bboxes[:, ij])
                bboxes_i.append(interpfn(frame_i))
            bboxes_i = np.stack(bboxes_i, axis=1)

            if (
                max(np.mean(bboxes_i[:, 2] - bboxes_i[:, 0]), np.mean(bboxes_i[:, 3] - bboxes_i[:, 1]))
                > opt.min_face_size
            ):
                tracks.append({"frame": frame_i, "bbox": bboxes_i})

    return tracks


def crop_video(track: Dict, images_path: str, audio_path: str, cropped_video_path: str, opt: SyncNetConfig) -> Dict:
    tmp_video_path, tmp_audio_path = f"{opt.tmp_dir}/temp_video.mov", f"{opt.tmp_dir}/temp_audio.wav"
    cs = opt.crop_scale

    dets = {"x": [], "y": [], "s": []}
    for det in track["bbox"]:
        dets["s"].append(max((det[3] - det[1]), (det[2] - det[0])) / 2)
        dets["y"].append((det[1] + det[3]) / 2)  # crop center x
        dets["x"].append((det[0] + det[2]) / 2)  # crop center y
    dets["s"] = signal.medfilt(dets["s"], kernel_size=13)
    dets["x"] = signal.medfilt(dets["x"], kernel_size=13)
    dets["y"] = signal.medfilt(dets["y"], kernel_size=13)

    flist = glob.glob(f"{images_path}/*.jpg")
    flist.sort()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vOut = cv2.VideoWriter(tmp_video_path, fourcc, opt.frame_rate, (224, 224))
    for fidx, frame in enumerate(track["frame"]):
        bs = dets["s"][fidx]  # Detection box size
        bsi = int(bs * (1 + 2 * cs))  # Pad videos by this amount

        image = cv2.imread(flist[frame])
        frame = np.pad(image, ((bsi, bsi), (bsi, bsi), (0, 0)), "constant", constant_values=(110, 110))
        my = dets["y"][fidx] + bsi  # BBox center Y
        mx = dets["x"][fidx] + bsi  # BBox center X

        face = frame[int(my - bs) : int(my + bs * (1 + 2 * cs)), int(mx - bs * (1 + cs)) : int(mx + bs * (1 + cs))]
        vOut.write(cv2.resize(face, (224, 224)))
    vOut.release()

    # ========== CROP AUDIO FILE ==========
    audiostart = (track["frame"][0]) / opt.frame_rate
    audioend = (track["frame"][-1] + 1) / opt.frame_rate
    crop_audio(audio_path, audiostart, audioend, tmp_audio_path)

    # ========== COMBINE AUDIO AND VIDEO FILES ==========
    combine_video_and_audio(tmp_video_path, tmp_audio_path, cropped_video_path)
    os.remove(tmp_video_path)
    os.remove(tmp_audio_path)

    log.debug(f"Written {cropped_video_path}")
    log.info(f"Mean pos: x {np.mean(dets['x']):.2f} y {np.mean(dets['y']):.2f} s {np.mean(dets['s']):.2f}")

    return {"track": track, "proc_track": dets}


def face_detection(img_folder: str, s3fd: S3FD, opt: SyncNetConfig) -> List[np.array]:
    flist = glob.glob(f"{img_folder}/*.jpg")
    flist.sort()

    dets = []
    for idx, fname in tqdm(enumerate(flist), total=len(flist), desc="Face detection"):
        image = cv2.imread(fname)
        image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxes = s3fd.detect_faces(image_np, conf_th=0.9, scales=[opt.facedet_scale])
        frame_dets = [{"frame": idx, "bbox": (bbox[:-1]).tolist(), "conf": bbox[-1]} for bbox in bboxes]
        dets.append(frame_dets)

    return dets


def scene_detection(video_path) -> List[Tuple[FrameTimecode, FrameTimecode]]:
    video_manager = VideoManager([video_path])
    base_timecode = video_manager.get_base_timecode()
    video_manager.set_downscale_factor()
    video_manager.start()

    stats_manager = StatsManager()
    scene_manager = SceneManager(stats_manager)
    # Add ContentDetector algorithm (constructor takes detector options like threshold).
    scene_manager.add_detector(ContentDetector())
    scene_manager.detect_scenes(frame_source=video_manager)

    scene_list = scene_manager.get_scene_list(base_timecode)
    if len(scene_list) == 0:
        scene_list = [(video_manager.get_base_timecode(), video_manager.get_current_timecode())]

    return scene_list
