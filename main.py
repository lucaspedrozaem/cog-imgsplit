# main.py
# ---------------------------------------------------------------
# Utilities:
#   – download_file()         : HTTP/HTTPS download helper
#   – combine_video_and_logo(): trim listing video (optional),
#                               then append white outro with fade-in logo
# ---------------------------------------------------------------

import os
import mimetypes
import tempfile
import requests
from urllib.parse import urlparse
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

    # Pick extension
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
    video_path: str,
    logo_path: str,
    output_path: str,
    #
    # NEW ➟ optional trimming
    #
    video_duration: float | None = None,  # seconds to KEEP from start
    outro_duration: float = 3.0,
    fade_duration: float = 1.5,
    logo_rel_width: float = 0.30,
    target_resolution: tuple[int, int] | None = None,
    keep_audio: bool = True,
) -> None:
    """
    Trim the input video to `video_duration` seconds (if provided),
    then append a white canvas outro with a fading-in logo.
    """
    # 1) Load main clip
    main_clip = VideoFileClip(video_path)

    # Optional trim/pad
    if video_duration and video_duration > 0:
        if main_clip.duration >= video_duration:
            main_clip = main_clip.subclip(0, video_duration)
        else:
            # If shorter, simply leave it; you could pad with freeze-frame if desired.
            pass

    # Optional resize
    if target_resolution:
        main_clip = main_clip.resize(newsize=target_resolution)

    # Remove audio if requested
    if not keep_audio:
        main_clip = main_clip.without_audio()

    w, h = main_clip.size
    fps = main_clip.fps or 30

    # 2) Build outro
    bg = ColorClip(size=(w, h), color=(255, 255, 255), duration=outro_duration)

    logo = (
        ImageClip(logo_path, transparent=True)
        .resize(width=int(w * logo_rel_width))
        .set_duration(outro_duration)
        .set_pos("center")
        .fx(vfx.fadein, fade_duration)
    )

    outro = CompositeVideoClip([bg, logo])

    # 3) Concatenate and export
    final = concatenate_videoclips([main_clip, outro], method="compose")

    final.write_videofile(
        output_path,
        fps=fps,
        codec="libx264",
        audio_codec="aac" if keep_audio else None,
        preset="medium",
        threads=4,
    )

    # 4) Clean up resources
    main_clip.close()
    outro.close()
    final.close()


# ---------------------------------------------------------------
# Optional CLI for local testing
#   python main.py listing.mp4 logo.png output.mp4 7
# ---------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if not (4 <= len(sys.argv) <= 5):
        print(
            "Usage: python main.py <video.mp4> <logo.png> <output.mp4> "
            "[trim_seconds]"
        )
        sys.exit(1)

    combine_video_and_logo(
        video_path=sys.argv[1],
        logo_path=sys.argv[2],
        output_path=sys.argv[3],
        video_duration=float(sys.argv[4]) if len(sys.argv) == 5 else None,
    )
