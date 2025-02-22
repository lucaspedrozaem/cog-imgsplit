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

    # Draw background (rounded rectangle)
    if background_on:
        bg_color = (0, 0, 0, int(background_opacity * 255))  # Semi-transparent black
        draw.rounded_rectangle(
            [(0, 0), (img_width, img_height)],
            fill=bg_color,
            radius=border_radius
        )

    # Compute offsets to center text
    offset_x = (img_width - text_width) / 2 - x0
    offset_y = (img_height - text_height) / 2 - y0

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


def smoothstep(u):
    """Smoothstep easing function for u in [0,1]."""
    return 3*u**2 - 2*u**3

def apply_ken_burns_effect(image_clip: mpe.ImageClip, duration: float) -> mpe.VideoClip:
    """
    Apply a simple Ken Burns effect (one zoom+pan transition) using subpixel-accurate extraction.
    
    - Starts at (zoom=1.0, offset=0)
    - Ends at (zoom=Z, offsetX) with a guaranteed visible movement
    - Uses smoothstep interpolation to avoid abrupt changes
    - Clamps the crop so it never goes out of bounds
    """
    w, h = image_clip.size

    # Pick a stronger final zoom (e.g. 1.2 to 1.4) to ensure a noticeable effect
    final_zoom = 1.3  # Fixed for a clear result; feel free to randomize (e.g. random.uniform(1.2, 1.4))

    # Choose a noticeable horizontal offset (e.g., 20% of image width)
    # Positive = pan to the right, negative = pan to the left
    offsetX = 0.2 * w  # e.g., 20% of width
    # Try negative if you prefer leftward: offsetX = -0.2 * w
    
    # We'll define exactly 2 key states, so there's just one phase
    key_states = [
        (1.0, 0.0),       # start: full view, no offset
        (final_zoom, offsetX)  # end: zoomed and horizontally offset
    ]
    
    n_phases = 1
    # Just one phase = entire duration
    durations = [duration]
    total_effect_duration = duration
    
    # Set the clip duration
    image_clip = image_clip.set_duration(total_effect_duration)
    boundaries = [0, total_effect_duration]

    def advanced_effect(get_frame, t):
        # Because there's only 1 phase, the logic is straightforward
        phase_index = 0
        t_phase = t - boundaries[phase_index]
        u = t_phase / durations[phase_index]
        u = max(0, min(u, 1))
        u_eased = smoothstep(u)

        # Interpolate between (1.0, 0.0) and (final_zoom, offsetX)
        start_zoom, start_offset = key_states[0]
        end_zoom, end_offset = key_states[1]
        current_zoom = start_zoom + (end_zoom - start_zoom) * u_eased
        current_offset = start_offset + (end_offset - start_offset) * u_eased

        # Compute the crop size (float) from the current zoom
        crop_w = w / current_zoom
        crop_h = h / current_zoom
        
        # Center of the crop; shift horizontally by current_offset
        center_x = (w / 2) + current_offset
        center_y = h / 2
        
        # Clamp so the crop remains fully within the image
        center_x = max(crop_w/2, min(center_x, w - crop_w/2))
        center_y = max(crop_h/2, min(center_y, h - crop_h/2))
        
        # Subpixel extraction to avoid jitter
        patch = cv2.getRectSubPix(get_frame(t), (int(round(crop_w)), int(round(crop_h))), (center_x, center_y))
        resized = cv2.resize(patch, (w, h), interpolation=cv2.INTER_LINEAR)
        return resized

    return image_clip.fl(advanced_effect, apply_to=['mask', 'video'])



def sync_videos_to_song(video_info: list, song_file: str, do_trim: bool, output_file: str,
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
    DEFAULT_FONT_PATH = "dejavu-sans-bold.ttf"

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
            clip = apply_ken_burns_effect(base_clip, target_duration)
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
        temp_audiofile="temp-audio.m4a",
        remove_temp=True
    )
    for temp_file, _ in downloaded_inputs:
        try:
            os.remove(temp_file)
            print(f"Removed temporary file: {temp_file}")
        except Exception as e:
            print(f"Error removing temporary file {temp_file}: {e}")
