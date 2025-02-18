import os
from shutil import rmtree


class SyncNetConfig:
    def __init__(
        self,
        data_dir,
        facedet_scale=0.25,
        crop_scale=0.40,
        min_track=50,
        frame_rate=25,
        audio_sample_rate=16000,
        num_failed_det=25,
        min_face_size=100,
        s3fd_weights_path="./weights/sfd_face.pth",
        syncnet_weights_path="./weights/syncnet_v2.model",
        batch_size=20,
        vshift=15,
        device="cpu",
    ):
        self.facedet_scale = facedet_scale
        self.crop_scale = crop_scale
        self.min_track = min_track
        self.frame_rate = frame_rate
        self.audio_sample_rate = audio_sample_rate
        self.num_failed_det = num_failed_det
        self.min_face_size = min_face_size

        self.s3fd_weights_path = s3fd_weights_path
        self.syncnet_weights_path = syncnet_weights_path
        self.batch_size = batch_size
        self.vshift = vshift
        self.device = device

        self.data_dir = data_dir
        self.tmp_dir = f"{self.data_dir}/tmp"
        self.crop_dir = f"{self.data_dir}/cropped_scenes"
        self.frames_dir = f"{self.data_dir}/all_frames"

        # ========== DELETE EXISTING AND MAKE NEW DIRECTORIES ==========
        for folder in [self.tmp_dir, self.crop_dir, self.frames_dir]:
            if os.path.exists(folder):
                rmtree(folder)
            os.makedirs(folder)
