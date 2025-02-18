import logging
import subprocess
from typing import List

log = logging.getLogger(__name__)


def call_ffmpeg(command: List[str], debug=False):
    log.debug(f"FFMpeg command: {' '.join(command)}")
    if not debug:
        return_code = subprocess.call(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        return_code = subprocess.call(command)
    if return_code != 0:
        raise RuntimeError(f"ffmpeg returned status {return_code}")


def change_fps(input_path: str, fps: int, output_path: str):
    # fmt: off
    command = [
        "ffmpeg", "-hide_banner", "-y",
        "-i", input_path,
        "-crf", "17",
        "-async", "1",
        "-r", str(fps),
        output_path,
    ]
    # fmt: on
    call_ffmpeg(command)


def extract_all_frames(input_path: str, output_path: str):
    # fmt: off
    command = [
        "ffmpeg", "-hide_banner", "-y",
        "-i", input_path,
        "-qscale:v", "2",
        "-threads", "1",
        "-f", "image2",
        output_path,
    ]
    # fmt: on
    call_ffmpeg(command)


def extract_scene_frames(input_path: str, output_path: str):
    # fmt: off
    command = [
        "ffmpeg", "-hide_banner", "-y",
        "-i", input_path,
        "-threads", "1",
        "-f", "image2",
        output_path
    ]
    # fmt: on
    call_ffmpeg(command)


def extract_all_audio(input_path: str, sample_rate: int, output_path: str):
    # fmt: off
    command = [
        "ffmpeg", "-hide_banner", "-y",
        "-i", input_path,
        "-ac", "1",
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        output_path
    ]
    # fmt: on
    call_ffmpeg(command)


def extract_scene_audio(input_path: str, sample_rate: int, output_path: str):
    # fmt: off
    command = [
        "ffmpeg", "-hide_banner", "-y",
        "-i", input_path,
        "-async", "1",
        "-ac", "1",
        "-vn",
        "-shortest",
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        output_path
    ]
    # fmt: on
    call_ffmpeg(command)


def crop_audio(input_path: str, ss: float, to: float, output_path: str):
    # ========== CROP AUDIO FILE ==========
    # fmt: off
    command = [
        "ffmpeg", "-hide_banner", "-y",
        "-i", input_path,
        "-ss", f"{ss}",
        "-to", f"{to}",
        output_path
    ]
    # fmt: on
    call_ffmpeg(command)


def combine_video_and_audio(input_video_path: str, input_audio_path: str, output_path: str):
    # ========== COMBINE AUDIO AND VIDEO FILES ==========
    # fmt: off
    command = [
        "ffmpeg", "-hide_banner", "-y",
        "-i", input_video_path,
        "-i", input_audio_path,
        "-c:v", "copy",
        "-c:a", "copy",
        output_path
    ]
    # fmt: on
    call_ffmpeg(command)
