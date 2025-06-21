# predict.py
# ---------------------------------------------------------------
# Cog predictor: optional listing video + white-logo outro render
# ---------------------------------------------------------------

import os
import tempfile
from typing import Optional, Tuple

from cog import BasePredictor, Input, Path

from main import combine_video_and_logo, download_file


class Predictor(BasePredictor):
    """Download assets, add branded outro, return an MP4."""

    def setup(self):
        pass

    def predict(
        self,
        video_url: str = Input(
            description=(
                "Public URL to the listing video (leave blank to render only "
                "the logo outro)."
            ),
            default="",
        ),
        logo_url: str = Input(
            description="Public URL to a transparent-background logo."
        ),
        video_duration: float = Input(
            description="Seconds to KEEP from listing video before outro (0 = full).",
            default=0.0,
            ge=0.0,
            le=1_800.0,
        ),
        outro_duration: float = Input(
            description="Seconds for the white canvas outro.",
            default=3.0,
            ge=0.5,
            le=10.0,
        ),
        fade_duration: float = Input(
            description="Seconds the logo takes to fade in.",
            default=1.5,
            ge=0.2,
            le=5.0,
        ),
        logo_rel_width: float = Input(
            description="Logo width as a fraction of video width (0-1).",
            default=0.30,
            ge=0.05,
            le=1.0,
        ),
        target_resolution: str = Input(
            description=(
                "Output resolution WIDTHxHEIGHT (e.g. '1920x1080'). "
                "Leave blank to keep original / default."
            ),
            default="",
        ),
        keep_audio: bool = Input(
            description="Keep original audio if listing video is provided.",
            default=True,
        ),
    ) -> Path:
        # 1) Download assets
        video_path: Optional[str] = None
        
        if video_url == "null":
            video_url = ""

        if video_url.strip():
            video_path = download_file(video_url)

        logo_path = download_file(logo_url)

        # 2) Parse resolution
        target_res: Optional[Tuple[int, int]] = None
        if target_resolution:
            try:
                w, h = map(int, target_resolution.lower().split("x"))
                target_res = (w, h)
            except Exception as e:
                raise ValueError(
                    "target_resolution must be WIDTHxHEIGHT, e.g. '1920x1080'."
                ) from e

        # 3) Temp output
        output_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name

        # 4) Combine
        combine_video_and_logo(
            video_path=video_path,                       # may be None
            logo_path=logo_path,
            output_path=output_file,
            video_duration=video_duration or None,
            outro_duration=outro_duration,
            fade_duration=fade_duration,
            logo_rel_width=logo_rel_width,
            target_resolution=target_res,
            keep_audio=keep_audio,
        )

        # 5) Clean-up
        if video_path:
            os.remove(video_path)
        os.remove(logo_path)

        return Path(output_file)
