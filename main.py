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

def create_caption_clip(text: str, duration: float, target_width: int, pos) -> mpe.VideoClip:
    """Create a caption clip (an ImageClip) with white text and a black stroke that fades in."""
    fontsize = 50
    stroke_width = 2
    font_path = get_font_path()
    font = ImageFont.truetype(font_path, fontsize) if font_path else ImageFont.load_default()
    
    # Use textbbox to compute text dimensions.
    dummy_img = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
    draw = ImageDraw.Draw(dummy_img)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    padding = 20
    img_width = text_width + 2 * padding
    img_height = text_height + 2 * padding
    image = Image.new("RGBA", (img_width, img_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    # Draw black stroke.
    for dx in range(-stroke_width, stroke_width+1):
        for dy in range(-stroke_width, stroke_width+1):
            draw.text((padding+dx, padding+dy), text, font=font, fill="black")
    # Draw white text.
    draw.text((padding, padding), text, font=font, fill="white")
    
    temp_image_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
    image.save(temp_image_path)
    
    caption_clip = mpe.ImageClip(temp_image_path).set_duration(duration)
    caption_clip = caption_clip.crossfadein(0.5)
    return caption_clip

def apply_ken_burns_effect(image_clip: mpe.ImageClip, duration: float) -> mpe.VideoClip:
    """Apply a Ken Burns effect (slow zoom and horizontal pan) to an ImageClip."""
    w, h = image_clip.size
    def ken_burns(get_frame, t):
        zoom = 1 + 0.2 * (t / duration)
        new_w, new_h = int(w / zoom), int(h / zoom)
        max_offset = (w - new_w) // 2
        offset_x = int(max_offset * (2 * t / duration - 1))
        offset_y = 0
        x1 = max(0, (w - new_w) // 2 + offset_x)
        y1 = max(0, (h - new_h) // 2 + offset_y)
        x2 = x1 + new_w
        y2 = y1 + new_h
        frame = get_frame(t)
        cropped = frame[y1:y2, x1:x2]
        resized = cv2.resize(cropped, (w, h))
        return resized
    return image_clip.fl(ken_burns, apply_to=['mask', 'video'])

def sync_videos_to_song(video_info: list, song_file: str, output_file: str,
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
    for video in video_info:
        if isinstance(video, dict):
            url = video.get("url")
            caption = video.get("caption")
        else:
            url = video
            caption = None
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
    
    # Beat detection using librosa.
    y, sr = librosa.load(song_file, sr=None)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    times_env = librosa.times_like(onset_env, sr=sr)
    threshold = np.median(onset_env) + np.std(onset_env)
    good_indices = np.where(onset_env > threshold)[0]
    if good_indices.size > 0:
        first_good_time = times_env[good_indices[0]]
        print(f"First good moment detected at {first_good_time:.2f}s")
        beat_times = beat_times[beat_times >= first_good_time]
    else:
        print("No significant onset detected; using all beat times.")
    
    if beat_times.size == 0:
        raise ValueError("No beats detected after the first good moment; check the song file.")
    
    song_audio = mpe.AudioFileClip(song_file)
    song_duration = song_audio.duration
    if song_duration - beat_times[-1] > 0.5:
        beat_times = np.append(beat_times, song_duration)
    
    print(f"Detected tempo: {float(tempo):.2f} BPM")
    print("Beat times:", beat_times)
    
    segments = []
    i = 0
    while i < len(beat_times) - 1 and len(segments) < allowed_segments:
        start_time = beat_times[i]
        target_dur = random.uniform(3, 5)
        j = i + 1
        while j < len(beat_times) and (beat_times[j] - start_time) < target_dur:
            j += 1
        if j < len(beat_times):
            if (beat_times[j] - start_time) <= 5 or (j - 1) == i:
                end_time = beat_times[j]
            else:
                end_time = beat_times[j - 1]
        else:
            end_time = beat_times[-1]
        segments.append((start_time, end_time))
        i = j
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
        speed_factor = orig_duration / target_duration
        clip = clip.fx(vfx.speedx, speed_factor)
        clip = clip.set_duration(target_duration)
        if caption:
            pos = random.choice([("left", "bottom"), ("right", "bottom")])
            caption_clip = create_caption_clip(caption, target_duration, target_resolution[0], pos)
            caption_clip = caption_clip.set_position(pos)
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
