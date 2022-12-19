import glob
import os
import pdb
import pickle
import subprocess
import time

import cv2
import numpy as np
from scenedetect.detectors import ContentDetector
from scenedetect.scene_manager import SceneManager
from scenedetect.stats_manager import StatsManager
from scenedetect.video_manager import VideoManager
from scipy import signal
from scipy.interpolate import interp1d
from scipy.io import wavfile

from syncnet.config import Config
from syncnet.s3fd.s3fd import S3FD


def bb_intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou


def track_shot(opt: Config, scenefaces):
    """
    face tracking
    """

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


def crop_video(opt, track, cropfile):
    flist = glob.glob(os.path.join(opt.frames_dir, opt.reference, "*.jpg"))
    flist.sort()

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    vOut = cv2.VideoWriter(cropfile + "t.avi", fourcc, opt.frame_rate, (224, 224))

    dets = {"x": [], "y": [], "s": []}

    for det in track["bbox"]:
        dets["s"].append(max((det[3] - det[1]), (det[2] - det[0])) / 2)
        dets["y"].append((det[1] + det[3]) / 2)  # crop center x
        dets["x"].append((det[0] + det[2]) / 2)  # crop center y

    # Smooth detections
    dets["s"] = signal.medfilt(dets["s"], kernel_size=13)
    dets["x"] = signal.medfilt(dets["x"], kernel_size=13)
    dets["y"] = signal.medfilt(dets["y"], kernel_size=13)

    for fidx, frame in enumerate(track["frame"]):
        cs = opt.crop_scale

        bs = dets["s"][fidx]  # Detection box size
        bsi = int(bs * (1 + 2 * cs))  # Pad videos by this amount

        image = cv2.imread(flist[frame])
        frame = np.pad(image, ((bsi, bsi), (bsi, bsi), (0, 0)), "constant", constant_values=(110, 110))
        my = dets["y"][fidx] + bsi  # BBox center Y
        mx = dets["x"][fidx] + bsi  # BBox center X

        face = frame[int(my - bs) : int(my + bs * (1 + 2 * cs)), int(mx - bs * (1 + cs)) : int(mx + bs * (1 + cs))]
        vOut.write(cv2.resize(face, (224, 224)))

    audiotmp = os.path.join(opt.tmp_dir, opt.reference, "audio.wav")
    audiostart = (track["frame"][0]) / opt.frame_rate
    audioend = (track["frame"][-1] + 1) / opt.frame_rate

    vOut.release()

    # ========== CROP AUDIO FILE ==========
    command = "ffmpeg -hide_banner -y -i %s -ss %.3f -to %.3f %s" % (
        os.path.join(opt.avi_dir, opt.reference, "audio.wav"),
        audiostart,
        audioend,
        audiotmp,
    )
    output = subprocess.call(command, shell=True, stdout=None)
    if output != 0:
        pdb.set_trace()

    sample_rate, audio = wavfile.read(audiotmp)

    # ========== COMBINE AUDIO AND VIDEO FILES ==========
    command = "ffmpeg -hide_banner -y -i %st.avi -i %s -c:v copy -c:a copy %s.avi" % (cropfile, audiotmp, cropfile)
    output = subprocess.call(command, shell=True, stdout=None)
    if output != 0:
        pdb.set_trace()

    print("Written %s" % cropfile)

    os.remove(cropfile + "t.avi")

    print("Mean pos: x %.2f y %.2f s %.2f" % (np.mean(dets["x"]), np.mean(dets["y"]), np.mean(dets["s"])))

    return {"track": track, "proc_track": dets}


def face_detection(opt, device):
    DET = S3FD(weights_path=opt.s3df_weights_path, device=device)

    flist = glob.glob(os.path.join(opt.frames_dir, opt.reference, "*.jpg"))
    flist.sort()

    dets = []
    for fidx, fname in enumerate(flist):
        start_time = time.time()

        image = cv2.imread(fname)
        image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxes = DET.detect_faces(image_np, conf_th=0.9, scales=[opt.facedet_scale])
        dets.append([])
        for bbox in bboxes:
            dets[-1].append({"frame": fidx, "bbox": (bbox[:-1]).tolist(), "conf": bbox[-1]})

        elapsed_time = time.time() - start_time

        print(
            "%s-%05d; %d dets; %.2f Hz"
            % (os.path.join(opt.avi_dir, opt.reference, "video.avi"), fidx, len(dets[-1]), (1 / elapsed_time))
        )

    savepath = os.path.join(opt.work_dir, opt.reference, "faces.pckl")
    with open(savepath, "wb") as fil:
        pickle.dump(dets, fil)

    return dets


def scene_detection(opt):
    video_manager = VideoManager([os.path.join(opt.avi_dir, opt.reference, "video.avi")])
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

    savepath = os.path.join(opt.work_dir, opt.reference, "scene.pckl")
    with open(savepath, "wb") as fil:
        pickle.dump(scene_list, fil)

    print("%s - scenes detected %d" % (os.path.join(opt.avi_dir, opt.reference, "video.avi"), len(scene_list)))

    return scene_list
