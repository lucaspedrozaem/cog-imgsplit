# predict.py
#
# Cog-style wrapper that:
#   1. Downloads a listing video and a transparent-background logo
#   2. Appends a white outro where the logo fades in
#   3. Returns the finished MP4
#
# Heavy lifting lives in main.combine_video_and_logo().
# ---------------------------------------------------------------------

import os
import tempfile
from cog import BasePredictor, Input, Path
from main import combine_video_and_logo, download_file


class Predictor(BasePredictor):
    """
    Appends a clean white canvas outro to a real-estate video
    and fades the agent’s logo in.
    """

    def setup(self):
        # Put any heavyweight, one-time initialisation here (GPU warm-up, etc.)
        pass

    def predict(
        self,
        video_url: str = Input(
            description="URL of the finished listing video (MP4, MOV, etc.).",
            default="https://example.com/listing.mp4",
        ),
        logo_url: str = Input(
            description="URL of the logo with transparent background (PNG/SVG/JPEG).",
            default="https://example.com/logo.png",
        ),
        outro_duration: float = Input(
            description="Length of the white outro canvas (seconds).",
            default=3.0,
            ge=0.5,
            le=10,
        ),
        fade_duration: float = Input(
            description="Time the logo takes to fade in (seconds).",
            default=1.5,
            ge=0.2,
            le=5,
        ),
        logo_rel_width: float = Input(
            description="Logo width as a fraction of the video width (0–1).",
            default=0.30,
            ge=0.05,
            le=1.0,
        ),
        target_resolution: str = Input(
            description=(
                "Optional output resolution in WIDTHxHEIGHT format, e.g. '1920x1080'. "
                "Leave blank to keep the original video resolution."
            ),
            default="",
        ),
        keep_audio: bool = Input(
            description="Preserve the original audio track if present.",
            default=True,
        ),
    ) -> Path:
        """Download inputs, run combine_video_and_logo(), return path to final MP4."""
        # ---------------------------------------------------------------------
        # 1) Download assets
        # ---------------------------------------------------------------------
        video_path = download_file(video_url)
        logo_path = download_file(logo_url)

        # ---------------------------------------------------------------------
        # 2) Parse optional resolution
        # ---------------------------------------------------------------------
        target_resolution_tuple = None
        if target_resolution:
            try:
                width, height = map(int, target_resolution.lower().split("x"))
                target_resolution_tuple = (width, height)
            except Exception as e:
                raise ValueError(
                    "target_resolution must be in WIDTHxHEIGHT format, e.g. '1920x1080'."
                ) from e

        # ---------------------------------------------------------------------
        # 3) Temp file for the finished video
        # ---------------------------------------------------------------------
        output_file = tempfile.NamedTemporaryFile(
            delete=False, suffix=".mp4"
        ).name

        # ---------------------------------------------------------------------
        # 4) Combine video + logo
        # ---------------------------------------------------------------------
        combine_video_and_logo(
            video_path=video_path,
            logo_path=logo_path,
            output_path=output_file,
            outro_duration=outro_duration,
            fade_duration=fade_duration,
            logo_rel_width=logo_rel_width,
            target_resolution=target_resolution_tuple,
            keep_audio=keep_audio,
        )

        # ---------------------------------------------------------------------
        # 5) Clean up downloads
        # ---------------------------------------------------------------------
        os.remove(video_path)
        os.remove(logo_path)

        # ---------------------------------------------------------------------
        # 6) Return path to final MP4
        # ---------------------------------------------------------------------
        return Path(output_file)
