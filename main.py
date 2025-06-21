# main.py
# ---------------------------------------------------------------
# Utilities:
#   – download_file()         : streamed HTTP/HTTPS download
#   – combine_video_and_logo(): optional trim, then append white
#                               outro with fading logo
# ---------------------------------------------------------------

import os
import mimetypes
import tempfile
import requests
from urllib.parse import urlparse
from typing import Optional, Tuple

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

    # Extension from URL → MIME → .bin fallback
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
# Core video-processing
# ---------------------------------------------------------------
def combine_video_and_logo(
    *,
    video_path: Optional[str],                 # ← may be None
    logo_path: str,
    output_path: str,
    video_duration: Optional[float] = None,    # seconds to KEEP
    outro_duration: float = 3.0,
    fade_duration: float = 1.5,
    logo_rel_width: float = 0.30,
    target_resolution: Optional[Tuple[int, int]] = None,
    keep_audio: bool = True,
) -> None:
    """
    If `video_path` is provided → trim / resize then append white outro.
    If `video_path` is None → render only the white-logo outro.
    """
    clips = []

    # ---------------------------- main listing video (optional)
    if video_path:
        listing = VideoFileClip(video_path)

        # Trim
        if video_duration and video_duration > 0 and listing.duration >= video_duration:
            listing = listing.subclip(0, video_duration)

        # Resize
        if target_resolution:
            listing = listing.resize(newsize=target_resolution)

        # Audio
        if not keep_audio:
            listing = listing.without_audio()

        clips.append(listing)
        base_w, base_h = listing.size
        fps = listing.fps or 30
    else:
        # No listing video supplied
        if target_resolution:
            base_w, base_h = target_resolution
        else:
            base_w, base_h = (1920, 1080)
        fps = 30

    # ---------------------------- build white outro + fading logo
    bg = ColorClip(size=(base_w, base_h), color=(255, 255, 255), duration=outro_duration)

    logo = (
        ImageClip(logo_path, transparent=True)
        .resize(width=int(base_w * logo_rel_width))
        .set_duration(outro_duration)
        .set_pos("center")
        .fx(vfx.fadein, fade_duration, initial_color=(255, 255, 255))
    )

    outro_clip = CompositeVideoClip([bg, logo])
    clips.append(outro_clip)

    # ---------------------------- concatenate & export
    final = clips[0] if len(clips) == 1 else concatenate_videoclips(clips, method="compose")

    final.write_videofile(
        output_path,
        fps=fps,
        codec="libx264",
        audio_codec="aac" if keep_audio and video_path else None,
        preset="medium",
        threads=4,
    )

    # Cleanup
    outro_clip.close()
    final.close()
    if video_path:
        listing.close()


# ---------------------------------------------------------------
# CLI for local testing
# ---------------------------------------------------------------
if __name__ == "__main__":
    import sys

    usage = (
        "Usage:\n"
        "  python main.py <logo.png> <output.mp4>                   # logo-only outro\n"
        "  python main.py <video.mp4> <logo.png> <output.mp4>        # full flow\n"
        "  python main.py <video.mp4> <logo.png> <output.mp4> <trim_seconds>"
    )

    if len(sys.argv) not in (3, 4, 5):
        print(usage)
        sys.exit(1)

    if len(sys.argv) == 3:  # logo only
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
