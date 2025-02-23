# main.py

import os
import random
import tempfile
from urllib.parse import urlparse

import cv2
import numpy as np
import librosa
from PIL import Image, ImageDraw, ImageFont
import moviepy.editor as mpe
import moviepy.video.fx.all as vfx
import requests

def download_file(url: str) -> str:
    """Download a file (video or image) from a URL into a temporary file."""
    parsed_url = urlparse(url)
    suffix = os.path.splitext(parsed_url.path)[1]
    if not suffix:
        suffix = ".jpg"
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(temp_file.name, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    return temp_file.name

def crop_to_aspect(clip: mpe.VideoClip, desired_aspect_ratio: float) -> mpe.VideoClip:
    """Center-crop the clip using a cover strategy so that its aspect ratio matches the desired value."""
    clip_w = int(clip.w)
    clip_h = int(clip.h)
    current_ratio = clip_w / clip_h

    if current_ratio > desired_aspect_ratio:
        new_width = int(round(desired_aspect_ratio * clip_h))
        if new_width % 2:
            new_width -= 1
        x_center = clip_w / 2
        x1 = int(x_center - new_width / 2)
        x2 = int(x_center + new_width / 2)
        return clip.crop(x1=x1, x2=x2)
    elif current_ratio < desired_aspect_ratio:
        new_height = int(round(clip_w / desired_aspect_ratio))
        if new_height % 2:
            new_height -= 1
        y_center = clip_h / 2
        y1 = int(y_center - new_height / 2)
        y2 = int(y_center + new_height / 2)
        return clip.crop(y1=y1, y2=y2)
    else:
        return clip

def zoom_to_fill(clip: mpe.VideoClip, target_resolution: tuple) -> mpe.VideoClip:
    """Scale up the clip (keeping its aspect ratio) until it fills the target resolution, then center-crop."""
    target_width, target_height = target_resolution
    scale_factor = max(target_width / clip.w, target_height / clip.h)
    resized_clip = clip.resize(scale_factor)
    return resized_clip.crop(x_center=resized_clip.w/2, 
                             y_center=resized_clip.h/2,
                             width=target_width, height=target_height)

def get_font_path() -> str:
    """Return a path to a common bold font if available."""
    dejavu = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    if os.path.exists(dejavu):
        return dejavu
    else:
        return ""  # If empty, Pillow will fall back to a default font

def create_caption_clip(
    text: str,
    duration: float,
    video_size: tuple[int, int],
    position: str = "bottom-center",
    # Text modifications
    uppercase: bool = False,
    font_path: str = None,  # e.g. "./fonts/Roboto-Regular.ttf"
    font_size: int = 50,
    text_color: str = "#FFFFFF",
    # Stroke (outline)
    stroke_size: int = 0,
    stroke_color: str = "#000000",
    # Background box
    background_on: bool = True,
    background_opacity: float = 0.5,
    border_radius: int=10,
    # Appearance
    padding: int = 20,
    hex_color: str = "#000000",
    fadein_duration: float = 0.5
) -> mpe.ImageClip:
    """
    Create a caption ImageClip with customizable text, stroke, background box,
    and position. The clip is crossfaded in over `fadein_duration`.
    """

    if uppercase:
        text = text.upper()

    # Load custom font if specified, else fallback
    if font_path:
        font = ImageFont.truetype(font_path, font_size)
    else:
        font = ImageFont.load_default()

    # Measure the text's bounding box relative to (0,0)
    dummy_img = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    draw_dummy = ImageDraw.Draw(dummy_img)
    x0, y0, x1, y1 = draw_dummy.textbbox((0, 0), text, font=font)

    text_width = x1 - x0
    text_height = y1 - y0

    # Determine total image size (text size + padding)
    img_width = text_width + 2 * padding
    img_height = text_height + 2 * padding

    # Create the final image
    image = Image.new("RGBA", (img_width, img_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    def hex_to_rgba(hex_color, opacity=1.0):
        """Convert hex color to an RGBA tuple with adjustable opacity."""
        hex_color = hex_color.lstrip('#')
        r, g, b = [int(hex_color[i:i+2], 16) for i in (0, 2, 4)]
        a = int(opacity * 255)
        return (r, g, b, a)

    # Draw background (rounded rectangle)
    if background_on:
        #bg_color = (0, 0, 0, int(background_opacity * 255))  # Semi-transparent black
        bg_color = hex_to_rgba(hex_color, background_opacity)
        draw.rounded_rectangle(
            [(0, 0), (img_width, img_height)],
            fill=bg_color,
            radius=border_radius
        )

    # Compute offsets to center text
    offset_x = (img_width - text_width) / 2 - x0
    offset_y = (img_height - text_height) / 2 - y0

    vertical_shift = 2
    offset_y += vertical_shift

    # Draw stroke by offsetting text in a small grid
    if stroke_size > 0:
        for dx in range(-stroke_size, stroke_size + 1):
            for dy in range(-stroke_size, stroke_size + 1):
                draw.text(
                    (offset_x + dx, offset_y + dy),
                    text,
                    font=font,
                    fill=stroke_color
                )

    # Draw main text
    draw.text((offset_x, offset_y), text, font=font, fill=text_color)

    # Save to a temporary file so we can load it as a MoviePy ImageClip
    temp_image_path = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
    image.save(temp_image_path)


    # Create a MoviePy ImageClip
    caption_clip = mpe.ImageClip(temp_image_path).set_duration(duration)
    # Optionally fade in
    caption_clip = caption_clip.crossfadein(fadein_duration)

    # Now we compute the final (x, y) position as a static tuple
    # rather than passing a lambda. This avoids the numpy.float64 error.
    W, H = video_size
    clip_w, clip_h = caption_clip.size
    margin = 20

    # A small helper function that returns (x, y) for each position
    def calc_position(pos: str):
        if pos == "top-left":
            return (margin, margin)
        elif pos == "top-center":
            return ((W - clip_w) // 2, margin)
        elif pos == "top-right":
            return (W - clip_w - margin, margin)
        elif pos == "bottom-left":
            return (margin, H - clip_h - margin)
        elif pos == "bottom-center":
            return ((W - clip_w) // 2, H - clip_h - margin)
        elif pos == "bottom-right":
            return (W - clip_w - margin, H - clip_h - margin)
        elif pos == "middle":
            return ((W - clip_w) // 2, (H - clip_h) // 2)
        else:
            # default to bottom-center
            return ((W - clip_w) // 2, H - clip_h - margin)

    final_xy = calc_position(position)

    # Set that static position
    caption_clip = caption_clip.set_position(final_xy)

    return caption_clip


# def apply_ken_burns_effect(image_clip: mpe.ImageClip, duration: float) -> mpe.VideoClip:
#     """Apply a Ken Burns effect (slow zoom and horizontal pan) to an ImageClip."""
#     w, h = image_clip.size
#     def ken_burns(get_frame, t):
#         zoom = 1 + 0.2 * (t / duration)
#         new_w, new_h = int(w / zoom), int(h / zoom)
#         max_offset = (w - new_w) // 2
#         offset_x = int(max_offset * (2 * t / duration - 1))
#         offset_y = 0
#         x1 = max(0, (w - new_w) // 2 + offset_x)
#         y1 = max(0, (h - new_h) // 2 + offset_y)
#         x2 = x1 + new_w
#         y2 = y1 + new_h
#         frame = get_frame(t)
#         cropped = frame[y1:y2, x1:x2]
#         resized = cv2.resize(cropped, (w, h))
#         return resized
#     return image_clip.fl(ken_burns, apply_to=['mask', 'video'])


import random
import numpy as np
import cv2
from moviepy.editor import VideoClip, ImageClip

def smoothstep(u: float) -> float:
    """Smoothstep easing function for u in [0,1]."""
    return 3*u**2 - 2*u**3

def apply_ken_burns_effect(
    image_clip: ImageClip,
    duration: float,
    start_hold: float = 0.5
) -> VideoClip:
    """
    A Ken Burns-style effect that chooses a sequence of zoom/pan states and
    smooth-transitions between them over 'duration' seconds.

    Features:
      - If duration < 6, it can randomly pick 2-phase or 1-phase sequences.
      - For longer durations, it picks from several multi-phase sequences.
      - Safe transitions ensure we never zoom out from full view directly.
      - We also try to avoid consecutive pans in the same direction.
      - We avoid immediate zoom in→out (or out→in) with a post-processing pass.
      - We optionally reduce the first phase to 'start_hold' seconds,
        ensuring the effect starts quickly rather than lingering at the start.

    :param image_clip: MoviePy ImageClip (still image).
    :param duration: total Ken Burns effect duration in seconds.
    :param start_hold: how long the first phase can be if it’s an “idle” start,
                       making the movement begin earlier (defaults to 0.5s).
    :return: A VideoClip with the Ken Burns effect applied.
    """
    # ----------------------
    # 1) Basic setup
    # ----------------------
    phase_min = 2
    w, h = image_clip.size

    # Random zoom factor for the 'interesting' zoom phases.
    Z = random.uniform(1.2, 1.4)

    # Calculate max horizontal offset for the given zoom.
    new_w_at_Z = w / Z
    max_possible_offset = (w - new_w_at_Z) / 2.0

    # Offsets for slide (zoomed in)
    slide_offset_min = 0.5 * max_possible_offset
    slide_offset_max = max_possible_offset

    # Smaller offset for pre-slide at full view
    pre_max = max(5, 0.05 * w)

    # ----------------------
    # 2) Pick a sequence of key states (zoom, offset)
    # ----------------------
    if duration < 6:
        # Sequences for short videos
        short_sequences = []

        # 2-phase sequence examples (3 key states)
        ss1 = [(1.0, 0.0), (Z, 0.0), (1.0, 0.0)]
        offset = random.uniform(slide_offset_min, slide_offset_max)
        if random.choice([True, False]):
            offset = -offset
        ss2 = [(1.0, 0.0), (Z, offset), (1.0, 0.0)]
        short_sequences.append(ss1)
        short_sequences.append(ss2)

        # 1-phase sequence (2 key states) - single movement
        offset = random.uniform(slide_offset_min, slide_offset_max)
        if random.choice([True, False]):
            offset = -offset
        ss_single = [(1.0, 0.0), (Z, offset)]
        short_sequences.append(ss_single)

        key_states = random.choice(short_sequences)

    else:
        # Sequences for longer videos
        long_sequences = []

        offset = random.uniform(slide_offset_min, slide_offset_max)
        if random.choice([True, False]):
            offset = -offset
        seqA = [(1.0, 0.0), (Z, 0.0), (Z, offset), (1.0, 0.0)]
        long_sequences.append(seqA)

        pre_offset = random.uniform(0, pre_max) * random.choice([-1, 1])
        seqB = [(1.0, 0.0), (1.0, pre_offset), (Z, pre_offset), (1.0, 0.0)]
        long_sequences.append(seqB)

        pre_offset = random.uniform(0, pre_max) * random.choice([-1, 1])
        offset = random.uniform(slide_offset_min, slide_offset_max)
        if random.choice([True, False]):
            offset = -offset
        seqC = [(1.0, 0.0), (1.0, pre_offset), (Z, pre_offset),
                (Z, offset), (1.0, 0.0)]
        long_sequences.append(seqC)

        offset1 = random.uniform(slide_offset_min, slide_offset_max) * random.choice([-1, 1])
        offset2 = random.uniform(slide_offset_min, slide_offset_max) * random.choice([-1, 1])
        seqD = [(1.0, 0.0), (Z, 0.0), (Z, offset1), (Z, offset2), (1.0, 0.0)]
        long_sequences.append(seqD)

        pre_offset = random.uniform(0, pre_max) * random.choice([-1, 1])
        offset = random.uniform(slide_offset_min, slide_offset_max)
        if random.choice([True, False]):
            offset = -offset
        seqE = [(1.0, 0.0), (1.0, pre_offset), (Z, pre_offset),
                (Z, offset), (Z, pre_offset), (1.0, 0.0)]
        long_sequences.append(seqE)

        key_states = random.choice(long_sequences)

    # ----------------------
    # 3) Safe transitions: never zoom out from full, alternate consecutive pans
    # ----------------------
    # 3.1) Ensure we don't do "full view -> full view" with offset
    for i in range(1, len(key_states)):
        next_zoom, next_offset = key_states[i]
        if next_zoom == 1.0 or next_offset != 0:
            prev_zoom, prev_offset = key_states[i-1]
            if prev_zoom == 1.0:
                # Force previous to be zoomed to avoid "zoom out from full" or no-op
                key_states[i-1] = (Z, prev_offset)

    # 3.2) Alternate consecutive pans
    for i in range(1, len(key_states)):
        prev_offset = key_states[i-1][1]
        cur_offset = key_states[i][1]
        if prev_offset != 0 and cur_offset != 0:
            # If they have the same sign, flip the current
            if (prev_offset > 0 and cur_offset > 0) or (prev_offset < 0 and cur_offset < 0):
                key_states[i] = (key_states[i][0], -cur_offset)

    # ----------------------
    # 4) Avoid immediate zoom in->out or out->in by merging states
    # ----------------------
    def zoom_direction(z1, z2):
        if z2 > z1: return "IN"
        elif z2 < z1: return "OUT"
        else: return "NONE"

    i = 0
    while i < len(key_states) - 2:
        z1 = key_states[i][0]
        z2 = key_states[i+1][0]
        z3 = key_states[i+2][0]
        d1 = zoom_direction(z1, z2)
        d2 = zoom_direction(z2, z3)
        # Check for back-to-back "IN -> OUT" or "OUT -> IN"
        if (d1 == "IN" and d2 == "OUT") or (d1 == "OUT" and d2 == "IN"):
            # Remove the middle state to avoid immediate reversal
            key_states.pop(i+1)
            # Step back one index if possible
            if i > 0: i -= 1
        else:
            i += 1

    # ----------------------
    # 5) Compute phase durations
    # ----------------------
    # # of phases is one less than # of key states
    n_phases = len(key_states) - 1
    if n_phases <= 0:
        # Edge case: if there's only 1 state, no transition, just set fixed duration
        image_clip = image_clip.set_duration(duration)
        return image_clip

    required_min = phase_min * n_phases
    if duration < required_min:
        # Compress all phases equally
        phase_durations = [duration / n_phases] * n_phases
        total_duration = duration
    else:
        extra = duration - required_min
        partition = np.random.dirichlet(np.ones(n_phases)) * extra
        phase_durations = [phase_min + part for part in partition]
        total_duration = sum(phase_durations)

    # ----------------------
    # 6) Optionally reduce the first phase (start_hold) so effect starts earlier
    # ----------------------
    # Only if there's more than 1 phase and the first phase is bigger than start_hold
    if n_phases > 1 and phase_durations[0] > start_hold:
        diff = phase_durations[0] - start_hold
        phase_durations[0] = start_hold
        # Distribute 'diff' among remaining phases
        for i in range(1, n_phases):
            phase_durations[i] += diff / (n_phases - 1)
        total_duration = sum(phase_durations)

    # Alternatively, you could remove an initial no-op state:
    # if len(key_states) > 1 and key_states[0] == key_states[1]:
    #     key_states.pop(0)
    #     n_phases = len(key_states) - 1
    #     # re-compute durations, etc.

    print("Key States (zoom, offset):", key_states)
    print("Phase durations:", ["{:.2f}".format(d) for d in phase_durations])
    print("Total effect duration: {:.2f} seconds".format(total_duration))

    image_clip = image_clip.set_duration(total_duration)

    # Prepare phase boundaries
    boundaries = [0]
    for d in phase_durations:
        boundaries.append(boundaries[-1] + d)

    # ----------------------
    # 7) The Ken Burns frame function
    # ----------------------
    def ken_burns(get_frame, t):
        # 7.1) Identify current phase
        for j in range(n_phases):
            if boundaries[j] <= t < boundaries[j+1]:
                phase_index = j
                break
        else:
            phase_index = n_phases - 1

        # 7.2) Normalized time in [0,1] for this phase
        t_phase = t - boundaries[phase_index]
        phase_duration = phase_durations[phase_index]
        u = max(0, min(t_phase / phase_duration, 1))
        u_eased = smoothstep(u)

        # 7.3) Interpolate zoom & offset
        start_zoom, start_offset = key_states[phase_index]
        end_zoom, end_offset = key_states[phase_index+1]
        current_zoom = start_zoom + (end_zoom - start_zoom) * u_eased
        current_offset = start_offset + (end_offset - start_offset) * u_eased

        # 7.4) Compute crop
        crop_w = w / current_zoom
        crop_h = h / current_zoom

        center_x = w / 2 + current_offset
        center_y = h / 2

        # 7.5) Clamp center so we don't go out of bounds
        center_x = max(crop_w/2, min(center_x, w - crop_w/2))
        center_y = max(crop_h/2, min(center_y, h - crop_h/2))

        # 7.6) Extract and resize
        frame = get_frame(t)
        patch = cv2.getRectSubPix(
            frame,
            (int(round(crop_w)), int(round(crop_h))),
            (center_x, center_y)
        )
        resized = cv2.resize(patch, (w, h), interpolation=cv2.INTER_LINEAR)
        return resized

    # Apply the custom function
    return image_clip.fl(ken_burns, apply_to=['mask','video'])




def sync_videos_to_song(video_info: list, song_file: str, do_trim: bool, effect_hold: float, output_file: str,
                        loop_count: int = 1, aspect_ratio: str = "16:9",
                        target_resolution: tuple = None):
    """
    Processes inputs (videos and images) and synchronizes them to the beat of the song.
    
    Parameters:
      - video_info: List of dicts, each with "url" and optional "caption".
      - song_file: Local path to the song file.
      - output_file: Local path where the output video will be saved.
      - loop_count: Number of times to loop the input list.
      - aspect_ratio: Desired output aspect ratio (e.g. "16:9" or "9:16").
      - target_resolution: Tuple (width, height). Defaults to 1920x1080 for 16:9 or 1080x1920 for 9:16.
    """
    try:
        w_ratio, h_ratio = aspect_ratio.split(":")
        desired_ratio = float(w_ratio) / float(h_ratio)
    except Exception as e:
        raise ValueError("aspect_ratio must be in 'width:height' format, e.g. '16:9'.") from e

    if target_resolution is None:
        if aspect_ratio == "16:9":
            target_resolution = (1920, 1080)
        elif aspect_ratio == "9:16":
            target_resolution = (1080, 1920)
        else:
            target_resolution = (1920, 1080)
    
    print("Downloading input files...")
    downloaded_inputs = []
    # for video in video_info:
    #     if isinstance(video, dict):
    #         url = video.get("url")
    #         caption = video.get("caption")
    #     else:
    #         url = video
    #         caption = None
    #     try:
    #         local_path = download_file(url)
    #         downloaded_inputs.append((local_path, caption))
    #         print(f"Downloaded {url} -> {local_path}")
    #     except Exception as e:
    #         print(f"Failed to download {url}: {e}")

    POSSIBLE_POSITIONS = [
    "top-left", "top-center", "top-right",
    "bottom-left", "bottom-center", "bottom-right",
    "middle"
    ]

    # Example default font path if none is provided
    DEFAULT_FONT_PATH = "/fonts/Montserrat-VariableFont_wght.ttf"

    for video in video_info:
        if isinstance(video, dict):
            url = video.get("url")
            caption = video.get("caption")
            

            # Retrieve extra design variables from the dictionary.
            # If missing, use defaults (or random choice for position).
            position          = video.get("position", random.choice(POSSIBLE_POSITIONS))
            uppercase         = video.get("uppercase", True)
            font_path         = video.get("font_path", DEFAULT_FONT_PATH)
            font_size         = video.get("font_size", 50)
            text_color        = video.get("text_color", "#FFFFFF")
            stroke_size       = video.get("stroke_size", 10)
            border_radius     = video.get("border_radius", 10)
            stroke_color      = video.get("stroke_color", "#000000")
            background_on     = video.get("background_on", True)
            background_opacity= video.get("background_opacity", 0.5)
            slowdown          = video.get("slowdown", 1.0)
            padding           = video.get("padding", 10)
            bg_color          = video.get("bg_color", 10)
            hex_color         = video.get("hex_color", "#000000")
            fadein_duration   = video.get("fadein_duration", 0.5)

        else:
            # If 'video' is just a string, assume it's the URL.
            url = video
            caption = None

            # Use default design parameters in this scenario
            position          = random.choice(POSSIBLE_POSITIONS)
            uppercase         = True
            font_path         = DEFAULT_FONT_PATH
            font_size         = 50
            text_color        = "#FFFFFF"
            stroke_size       = 10
            stroke_color      = "#000000"
            background_on     = True
            background_opacity= 0.5
            border_radius     = 10
            padding           = 10
            slowdown = 1.0
            fadein_duration   = 0.5
            hex_color         = "#000000"
           

        try:
            local_path = download_file(url)
            downloaded_inputs.append((local_path, caption))
            print(f"Downloaded {url} -> {local_path}")
        except Exception as e:
            print(f"Failed to download {url}: {e}")


    if not downloaded_inputs:
        raise ValueError("No inputs were successfully downloaded.")
    
    all_inputs = downloaded_inputs * loop_count
    allowed_segments = len(all_inputs)
    
    # 1) Optionally trim up to 10s before the global peak.
    trim = do_trim  # or False

    # --- A) Find the global peak in the entire audio.
    y_full, sr_full = librosa.load(song_file, sr=None)
    onset_env_full = librosa.onset.onset_strength(y=y_full, sr=sr_full)
    times_env_full = librosa.times_like(onset_env_full, sr=sr_full)

    idx_peak_full = np.argmax(onset_env_full)
    peak_time_full = times_env_full[idx_peak_full]
    peak_intensity_full = onset_env_full[idx_peak_full]
    print(f"Global peak at {peak_time_full:.2f}s, intensity={peak_intensity_full:.2f}")

    # --- B) If trim=True, subclip from (peak_time_full - 10) or 0 if negative.
    if trim:
        cut_time = max(0, peak_time_full - 10)
        print(f"Trimming audio from {cut_time:.2f}s onward.")
        song_audio = mpe.AudioFileClip(song_file).subclip(cut_time, None)
        sub_duration = song_audio.duration

        # Reload that portion for beat detection.
        y, sr = librosa.load(song_file, sr=None, offset=cut_time, duration=sub_duration)
    else:
        print("No trimming; using entire track from 0s.")
        cut_time = 0
        song_audio = mpe.AudioFileClip(song_file)
        sub_duration = song_audio.duration
        y, sr = y_full, sr_full

    # 2) Beat detection on the chosen portion (trimmed or full).
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    # Compute onset env in the chosen portion.
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    times_env = librosa.times_like(onset_env, sr=sr)
    threshold = np.median(onset_env) + np.std(onset_env)

    # 3) Filter only good beats in [0..sub_duration].
    good_beats = []
    for bt in beat_times:
        idx_bt = np.argmin(np.abs(times_env - bt))
        if onset_env[idx_bt] > threshold:
            good_beats.append(bt)
    good_beats = np.array(good_beats)
    if good_beats.size == 0:
        raise ValueError("No good beats detected in the chosen audio.")

    # Append final boundary if leftover >0.5
    if sub_duration - good_beats[-1] > 0.5:
        good_beats = np.append(good_beats, sub_duration)

    print(f"Detected tempo: {float(tempo):.2f} BPM")
    print("Good beat times (in subclip timeline):", good_beats)

    # 4) Decide an 'ideal_dur' for uniform segments in [3..5].
    ideal_dur = random.uniform(3, 5)
    print(f"Ideal segment duration: {ideal_dur:.2f}s")

    # 5) Create segments so each segment is ~ideal_dur, 
    #    using good beats for boundaries. We stop if we reach allowed_segments.
    segments = []
    i = 0
    while i < len(good_beats) - 1 and len(segments) < allowed_segments:
        start_time = good_beats[i]
        end_goal = start_time + ideal_dur

        # Find the first beat >= end_goal
        j = i + 1
        while j < len(good_beats) and good_beats[j] < end_goal:
            j += 1
        if j < len(good_beats):
            end_time = good_beats[j]
            seg_dur = end_time - start_time
            # If seg_dur <3, try next beat if available
            if seg_dur < 3 and j+1 < len(good_beats):
                end_time = good_beats[j+1]
                j += 1
            segments.append((start_time, end_time))
            i = j
        else:
            # leftover
            if i < len(good_beats) - 1:
                segments.append((start_time, good_beats[-1]))
            break

    print("Segments (start, end):", segments)

    
    video_clips = []
    for idx, (seg_start, seg_end) in enumerate(segments):
        if idx >= allowed_segments:
            break
        target_duration = seg_end - seg_start
        local_path, caption = all_inputs[idx]
        print(f"Processing '{local_path}' for segment {idx+1} with target duration {target_duration:.2f}s")
        if local_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            base_clip = mpe.ImageClip(local_path).set_duration(target_duration)
            clip = apply_ken_burns_effect(base_clip, target_duration, effect_hold)
        else:
            clip = mpe.VideoFileClip(local_path).without_audio()
        clip = crop_to_aspect(clip, desired_ratio)
        clip = zoom_to_fill(clip, target_resolution)
        orig_duration = clip.duration
        #speed_factor = orig_duration / target_duration
        # Define slow_factor between 0 and 1 (1 = normal speed, 0.5 = half speed, etc.)
        slow_factor = slowdown  # Adjust this value (e.g., 0.5 for 50% speed)
        speed_factor = (orig_duration / target_duration) * slow_factor
        clip = clip.fx(vfx.speedx, speed_factor)
        clip = clip.set_duration(target_duration)

        if caption:

            #pos = random.choice([ "top-left", "top-center", "top-right", "bottom-left", "bottom-center", "bottom-right", "middle" ])
            
            caption_clip = create_caption_clip(
                text=caption,
                duration=target_duration,
                video_size=(target_resolution[0],target_resolution[1]),
                position=position,
                uppercase=uppercase,
                font_path=font_path,
                font_size=font_size,
                text_color=text_color,
                stroke_size=stroke_size,
                stroke_color=stroke_color,
                background_on=background_on,
                background_opacity=background_opacity,
                border_radius=border_radius,
                padding=10,
                hex_color=hex_color,
                fadein_duration=fadein_duration
            )

            clip = mpe.CompositeVideoClip([clip, caption_clip])

        video_clips.append(clip)
    
    final_video = mpe.concatenate_videoclips(video_clips, method="compose")
    final_duration = final_video.duration
    final_audio = song_audio.subclip(0, final_duration)
    final_video = final_video.set_audio(final_audio)
    final_video.write_videofile(
        output_file,
        codec="libx264",
        audio_codec="aac",
        fps=24,
        temp_audiofile="temp-audio.m4a",
        remove_temp=True,
    )
    for temp_file, _ in downloaded_inputs:
        try:
            os.remove(temp_file)
            print(f"Removed temporary file: {temp_file}")
        except Exception as e:
            print(f"Error removing temporary file {temp_file}: {e}")
