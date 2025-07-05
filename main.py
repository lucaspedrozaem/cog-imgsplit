# main.py
# ---------------------------------------------------------------
# Utilities:
#   – download_file()         : streamed HTTP/HTTPS download
#   – combine_video_and_logo(): optional trim, then append white
#                               outro with fading logo (PNG alpha kept)
# ---------------------------------------------------------------

import os
import mimetypes
import tempfile
import requests
from urllib.parse import urlparse
from typing import Optional, Tuple

import numpy as np
from PIL import Image                                  # ← NEW
from moviepy.editor import (
    VideoFileClip,
    ImageClip,
    ColorClip,
    CompositeVideoClip,
    concatenate_videoclips,
    vfx,
)

# ---------------------------------------------------------------
# HTTP download helper
# ---------------------------------------------------------------
def download_file(url: str, *, chunk_size: int = 8192, timeout: int = 30) -> str:
    """Stream a remote file to /tmp and return its local path."""
    try:
        resp = requests.get(url, stream=True, timeout=timeout)
        resp.raise_for_status()
    except Exception as exc:
        raise RuntimeError(f"Failed to download {url!r}: {exc}") from exc

    ext = os.path.splitext(urlparse(url).path)[1]
    if not ext:
        mime = resp.headers.get("content-type", "").split(";")[0].strip()
        ext = mimetypes.guess_extension(mime) or ".bin"

    tmp_path = tempfile.NamedTemporaryFile(
        suffix=ext, delete=False, prefix="download_"
    ).name

    with open(tmp_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)

    return tmp_path


# ---------------------------------------------------------------
# Helper: decide if a logo needs a dark background
# ---------------------------------------------------------------
def _logo_is_light(logo_path: str,
                   brightness_thresh: float = 0.8,
                   light_ratio_thresh: float = 0.7) -> bool:
    """
    Return True if the *visible* pixels of the PNG are mostly light
    (i.e. you'd lose them on a white canvas).
    """
    img = Image.open(logo_path).convert("RGBA")
    arr = np.asarray(img)
    alpha = arr[..., 3] > 20                 # ignore almost-transparent pixels
    if not alpha.any():
        return False                         # fully transparent → doesn't matter

    rgb = arr[alpha][..., :3] / 255.0
    # Per-Rec. 601 luma
    luma = (0.2126 * rgb[:, 0] +
            0.7152 * rgb[:, 1] +
            0.0722 * rgb[:, 2])
    return (luma > brightness_thresh).mean() > light_ratio_thresh


# ---------------------------------------------------------------
# updated: build an ImageClip that respects PNG transparency
# ---------------------------------------------------------------
def _make_logo_clip(
    logo_path: str,
    duration: float,
    width_px: int,
    fade_duration: float,
    bg_color: Tuple[int, int, int],
) -> ImageClip:
    """
    Read a PNG/SVG/JPEG, keep alpha if present, resize to width_px,
    and return an ImageClip that fades in from bg_color.
    """
    img = Image.open(logo_path).convert("RGBA")
    rgb = np.array(img)[..., :3]
    alpha = np.array(img)[..., 3] / 255.0

    logo_rgb = ImageClip(rgb).set_duration(duration)
    logo_mask = ImageClip(alpha, ismask=True).set_duration(duration)

    logo_clip = (
        logo_rgb.set_mask(logo_mask)
        .resize(width=width_px)
        .set_pos("center")
        .fx(vfx.fadein, fade_duration, initial_color=bg_color)
    )
    return logo_clip


# ---------------------------------------------------------------
# Core video-processing
# ---------------------------------------------------------------
def combine_video_and_logo(
    *,
    video_path: Optional[str],
    logo_path: str,
    output_path: str,
    video_duration: Optional[float] = None,
    outro_duration: float = 3.0,
    fade_duration: float = 1.5,
    logo_rel_width: float = 0.30,
    target_resolution: Optional[Tuple[int, int]] = None,
    keep_audio: bool = True,
    bg_color: Optional[Tuple[int, int, int]] = None,   # <-- new
) -> None:
    """
    If bg_color is None we'll analyse the logo:
        • mostly-light logo  → use black canvas (0,0,0)
        • otherwise          → use white canvas (255,255,255)
    """
    # Decide canvas colour ---------------------------------------------------
    if bg_color is None:
        bg_color = (0, 0, 0) if _logo_is_light(logo_path) else (255, 255, 255)

    clips = []

    # ---------------- main listing video (optional)
    if video_path:
        listing = VideoFileClip(video_path)

        if video_duration and video_duration > 0 and listing.duration >= video_duration:
            listing = listing.subclip(0, video_duration)

        if target_resolution:
            listing = listing.resize(newsize=target_resolution)

        if not keep_audio:
            listing = listing.without_audio()

        clips.append(listing)
        base_w, base_h = listing.size
        fps = listing.fps or 30
    else:
        base_w, base_h = target_resolution or (1920, 1080)
        fps = 30

    # ---------------- outro background + fading logo
    bg = ColorClip(size=(base_w, base_h), color=bg_color, duration=outro_duration)

    logo = _make_logo_clip(
        logo_path,
        duration=outro_duration,
        width_px=int(base_w * logo_rel_width),
        fade_duration=fade_duration,
        bg_color=bg_color,                # keep fade-in consistent
    )

    outro_clip = CompositeVideoClip([bg, logo])
    clips.append(outro_clip)

    # ---------------- concatenate & export
    final = clips[0] if len(clips) == 1 else concatenate_videoclips(clips, method="compose")

    final.write_videofile(
        output_path,
        fps=fps,
        codec="libx264",
        audio_codec="aac" if keep_audio and video_path else None,
        preset="medium",
        threads=4,
    )

    # tidy up
    outro_clip.close()
    final.close()
    if video_path:
        listing.close()



# ---------------------------------------------------------------
# CLI for quick local test
# ---------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) not in (3, 4, 5):
        print(
            "logo-only:\n"
            "  python main.py <logo.png> <output.mp4>\n\n"
            "listing + logo:\n"
            "  python main.py <video.mp4> <logo.png> <output.mp4> [trim_seconds]"
        )
        sys.exit(1)

    if len(sys.argv) == 3:
        combine_video_and_logo(
            video_path=None,
            logo_path=sys.argv[1],
            output_path=sys.argv[2],
        )
    else:
        combine_video_and_logo(
            video_path=sys.argv[1],
            logo_path=sys.argv[2],
            output_path=sys.argv[3],
            video_duration=float(sys.argv[4]) if len(sys.argv) == 5 else None,
        )
