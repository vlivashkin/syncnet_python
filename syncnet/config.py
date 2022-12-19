import os
from shutil import rmtree


class Config:
    def __init__(
        self,
        video_path,
        name,
        temp_dir="./temp",
        facedet_scale=0.25,
        crop_scale=0.40,
        min_track=100,
        frame_rate=25,
        num_failed_det=25,
        min_face_size=100,
        s3fd_weights_path="./weights/sfd_face.pth",
        syncnet_weights_path="./weights/syncnet_v2.model",
        batch_size=20,
        vshift=15,
    ):
        self.data_dir = temp_dir
        self.videofile = video_path
        self.reference = name
        self.facedet_scale = facedet_scale
        self.crop_scale = crop_scale
        self.min_track = min_track
        self.frame_rate = frame_rate
        self.num_failed_det = num_failed_det
        self.min_face_size = min_face_size

        self.s3fd_weights_path = s3fd_weights_path
        self.syncnet_weights_path = syncnet_weights_path
        self.batch_size = batch_size
        self.vshift = vshift

        self.avi_dir = os.path.join(self.data_dir, "pyavi")
        self.tmp_dir = os.path.join(self.data_dir, "pytmp")
        self.work_dir = os.path.join(self.data_dir, "pywork")
        self.crop_dir = os.path.join(self.data_dir, "pycrop")
        self.frames_dir = os.path.join(self.data_dir, "pyframes")

        # ========== DELETE EXISTING DIRECTORIES ==========
        if os.path.exists(os.path.join(self.work_dir, self.reference)):
            rmtree(os.path.join(self.work_dir, self.reference))
        if os.path.exists(os.path.join(self.crop_dir, self.reference)):
            rmtree(os.path.join(self.crop_dir, self.reference))
        if os.path.exists(os.path.join(self.avi_dir, self.reference)):
            rmtree(os.path.join(self.avi_dir, self.reference))
        if os.path.exists(os.path.join(self.frames_dir, self.reference)):
            rmtree(os.path.join(self.frames_dir, self.reference))
        if os.path.exists(os.path.join(self.tmp_dir, self.reference)):
            rmtree(os.path.join(self.tmp_dir, self.reference))

        # ========== MAKE NEW DIRECTORIES ==========
        os.makedirs(os.path.join(self.work_dir, self.reference))
        os.makedirs(os.path.join(self.crop_dir, self.reference))
        os.makedirs(os.path.join(self.avi_dir, self.reference))
        os.makedirs(os.path.join(self.frames_dir, self.reference))
        os.makedirs(os.path.join(self.tmp_dir, self.reference))
