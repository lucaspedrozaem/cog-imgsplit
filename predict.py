# predict.py
# ---------------------------------------------------------------
# Cog predictor: downloads listing video + logo, optionally trims
# the video, then appends a white outro where the logo fades in.
# ---------------------------------------------------------------

import os
import tempfile
from cog import BasePredictor, Input, Path
from typing import Optional, Tuple  # for type hints
from main import combine_video_and_logo, download_file


class Predictor(BasePredictor):
    """Downloads inputs, adds branded outro, returns MP4."""

    def setup(self):
        pass  # heavy initialisation could go here

    def predict(
        self,
        video_url: str = Input(description="Public URL to the listing video (mp4/mov)."),
        logo_url: str = Input(description="Public URL to a transparent-background logo."),
        #
        # Optional trim length →
        #
        video_duration: float = Input(
            description=(
                "Seconds to KEEP from the start of the input video BEFORE "
                "the outro is added. Set to 0 to keep the full video."
            ),
            default=0.0,
            ge=0.0,
            le=1800.0,  # 30 min max
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
            description="Logo width as a fraction of video width (0–1).",
            default=0.30,
            ge=0.05,
            le=1.0,
        ),
        target_resolution: str = Input(
            description=(
                "Optional output resolution WIDTHxHEIGHT (e.g. '1920x1080'). "
                "Leave blank to preserve original."
            ),
            default="",
        ),
        keep_audio: bool = Input(
            description="Keep the original audio track if present.",
            default=True,
        ),
    ) -> Path:
        # -------------------------------------------------------
        # 1) Download assets
        # -------------------------------------------------------
        video_path = download_file(video_url)
        logo_path = download_file(logo_url)

        # -------------------------------------------------------
        # 2) Parse optional resolution
        # -------------------------------------------------------
        target_res: Optional[Tuple[int, int]] = None
        if target_resolution:
            try:
                w, h = map(int, target_resolution.lower().split("x"))
                target_res = (w, h)
            except Exception as e:
                raise ValueError(
                    "target_resolution must be WIDTHxHEIGHT (e.g. '1920x1080')."
                ) from e

        # -------------------------------------------------------
        # 3) Prepare a temp output file
        # -------------------------------------------------------
        output_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name

        # -------------------------------------------------------
        # 4) Combine video + logo
        # -------------------------------------------------------
        combine_video_and_logo(
            video_path=video_path,
            logo_path=logo_path,
            output_path=output_file,
            video_duration=video_duration if video_duration > 0 else None,
            outro_duration=outro_duration,
            fade_duration=fade_duration,
            logo_rel_width=logo_rel_width,
            target_resolution=target_res,
            keep_audio=keep_audio,
        )

        # -------------------------------------------------------
        # 5) Clean up downloads
        # -------------------------------------------------------
        os.remove(video_path)
        os.remove(logo_path)

        return Path(output_file)
