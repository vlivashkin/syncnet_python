#!/usr/bin/python
# -*- coding: utf-8 -*-
# Video 25 FPS, Audio 16000HZ

import glob
import logging
import math
import os
import subprocess
import time
from shutil import rmtree
from typing import Tuple

import cv2
import numpy
import numpy as np
import python_speech_features
import torch
from scipy import signal
from scipy.io import wavfile

from syncnet.config import Config
from syncnet.syncnet_model import S

log = logging.getLogger(__name__)


# ==================== Get OFFSET ====================


def calc_pdist(feat1, feat2, vshift=10):
    win_size = vshift * 2 + 1
    feat2p = torch.nn.functional.pad(feat2, (0, 0, vshift, vshift))

    dists = []
    for i in range(0, len(feat1)):
        dists.append(
            torch.nn.functional.pairwise_distance(feat1[[i], :].repeat(win_size, 1), feat2p[i : i + win_size, :])
        )
    return dists


# ==================== MAIN DEF ====================


class SyncNetInstance(torch.nn.Module):
    def __init__(self, dropout=0, num_layers_in_fc_layers=1024, device="cuda:0"):
        super(SyncNetInstance, self).__init__()
        self.device = device
        self.__S__ = S(num_layers_in_fc_layers=num_layers_in_fc_layers).to(self.device)

    def load_parameters(self, path: str):
        loaded_state = torch.load(path, map_location=lambda storage, loc: storage)
        self_state = self.__S__.state_dict()
        for name, param in loaded_state.items():
            self_state[name].copy_(param)

    def evaluate(self, opt: Config, videofile: str) -> Tuple[np.array, np.array, np.array, np.array]:
        self.__S__.eval()

        # ========== ==========
        # Convert files
        # ========== ==========

        tmp_dir_ref = f"{opt.tmp_dir}/{opt.reference}"
        if os.path.exists(tmp_dir_ref):
            rmtree(tmp_dir_ref)
        os.makedirs(tmp_dir_ref)

        # fmt: off
        command = [
            "ffmpeg", "-hide_banner", "-y",
            "-i", videofile,
            "-threads", "1",
            "-f", "image2",
            f"{opt.tmp_dir}/{opt.reference}/%06d.jpg"
        ]
        # fmt: on
        returncode = subprocess.call(command, shell=True, stdout=None)
        assert returncode == 0

        # fmt: off
        command = [
            "ffmpeg", "-hide_banner", "-y",
            "-i", videofile,
            "-async", "1",
            "-ac", "1",
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            f"{opt.tmp_dir}/{opt.reference}/audio.wav"
        ]
        # fmt: on
        returncode = subprocess.call(command, shell=True, stdout=None)
        assert returncode == 0

        # ========== ==========
        # Load video
        # ========== ==========

        flist = glob.glob(f"{opt.tmp_dir}/{opt.reference}/*.jpg")
        flist.sort()

        images = []
        for fname in flist:
            images.append(cv2.imread(fname))

        im = numpy.stack(images, axis=3)
        im = numpy.expand_dims(im, axis=0)
        im = numpy.transpose(im, (0, 3, 4, 1, 2))

        imtv = torch.autograd.Variable(torch.from_numpy(im.astype(float)).float())

        # ========== ==========
        # Load audio
        # ========== ==========

        sample_rate, audio = wavfile.read(f"{opt.tmp_dir}/{opt.reference}/audio.wav")
        mfcc = zip(*python_speech_features.mfcc(audio, sample_rate))
        mfcc = numpy.stack([numpy.array(i) for i in mfcc])

        cc = numpy.expand_dims(numpy.expand_dims(mfcc, axis=0), axis=0)
        cct = torch.autograd.Variable(torch.from_numpy(cc.astype(float)).float())

        # ========== ==========
        # Check audio and video input length
        # ========== ==========

        audio_length, video_length = float(len(audio)) / 16000, float(len(images)) / 25
        if audio_length != video_length:
            log.info(f"WARNING: Audio ({audio_length:.4f}s) and video ({video_length:.4f}s) lengths are different.")

        min_length = min(len(images), math.floor(len(audio) / 640))

        # ========== ==========
        # Generate video and audio feats
        # ========== ==========

        lastframe = min_length - 5
        im_feat = []
        cc_feat = []

        tS = time.time()
        for i in range(0, lastframe, opt.batch_size):
            im_batch = [
                imtv[:, :, vframe : vframe + 5, :, :] for vframe in range(i, min(lastframe, i + opt.batch_size))
            ]
            im_in = torch.cat(im_batch, 0)
            im_out = self.__S__.forward_lip(im_in.to(self.device))
            im_feat.append(im_out.data.cpu())

            cc_batch = [
                cct[:, :, :, vframe * 4 : vframe * 4 + 20] for vframe in range(i, min(lastframe, i + opt.batch_size))
            ]
            cc_in = torch.cat(cc_batch, 0)
            cc_out = self.__S__.forward_aud(cc_in.to(self.device))
            cc_feat.append(cc_out.data.cpu())

        im_feat = torch.cat(im_feat, 0)
        cc_feat = torch.cat(cc_feat, 0)

        # ========== ==========
        # Compute offset
        # ========== ==========

        log.info(f"Compute time {time.time() - tS:.3f} sec.")

        dists = calc_pdist(im_feat, cc_feat, vshift=opt.vshift)
        mdist = torch.mean(torch.stack(dists, 1), 1)

        minval, minidx = torch.min(mdist, 0)

        offset = opt.vshift - minidx
        conf = torch.median(mdist) - minval

        fdist = numpy.stack([dist[minidx].numpy() for dist in dists])
        # fdist   = numpy.pad(fdist, (3,3), 'constant', constant_values=15)
        fconf = torch.median(mdist).numpy() - fdist
        fconfm = signal.medfilt(fconf, kernel_size=9)

        numpy.set_log.infooptions(formatter={"float": "{: 0.3f}".format})
        log.info("Framewise conf:")
        log.info(fconfm)
        log.info(f"AV offset:\t{offset}\nMin dist:\t{minval:.3f}\nConfidence:\t{conf:.3f}")

        dists_npy = numpy.array([dist.numpy() for dist in dists])
        return offset.numpy(), minval.numpy(), conf.numpy(), dists_npy
