# main.py

import os
import random
import tempfile
from urllib.parse import urlparse

import cv2
import numpy as np
import librosa
from PIL import Image, ImageDraw, ImageFont, ImageOps
import moviepy.editor as mpe
import moviepy.video.fx.all as vfx
from moviepy.audio.fx.all import audio_loop
import requests

IMAGE_EXTS = (
    '.jpg', '.jpeg', '.png', '.webp', '.bmp',
    '.tif', '.tiff', '.heic', '.heif', '.avif', '.gif'
)


def normalise_exif_orientation(path: str) -> str:
    """
    Return a path to an image whose pixel matrix has been rotated to match
    any EXIF Orientation tag.  Non-image files are returned unchanged.
    """
    if os.path.splitext(path.lower())[1] not in IMAGE_EXTS:
        return path                      # probably a video – leave untouched

    img = Image.open(path)
    img = ImageOps.exif_transpose(img)   # 🪄 auto-rotate & strip orientation tag

    # ---- NEW: strip alpha channel or pick a format that supports it ----
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")         # no more alpha ⇒ JPEG is fine

    tmp_path = tempfile.NamedTemporaryFile(
        delete=False, suffix=".jpg"      # keep life simple: always JPG out
    ).name
    img.save(tmp_path, format="JPEG", quality=95)

    return tmp_path

def download_file(url: str) -> str:
    """Download a file (video or image) from a URL into a temporary file."""
    parsed_url = urlparse(url)
    suffix = os.path.splitext(parsed_url.path)[1].lower()
    if not suffix:
        suffix = ".jpg"
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(temp_file.name, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    # # ---- 3) AVIF → PNG ------------------------------------------------------
    # if suffix == ".avif":
    #     png_path = tmp_path[:-5] + ".png"          # same name, new ext
    #     try:
    #         subprocess.run(
    #             ["ffmpeg", "-y", "-i", tmp_path, png_path],
    #             stdout=subprocess.DEVNULL,
    #             stderr=subprocess.DEVNULL,
    #             check=True,
    #         )
    #     except FileNotFoundError:  # ffmpeg not on PATH
    #         raise RuntimeError(
    #             "ffmpeg is required to convert AVIF images but was not found."
    #         ) from None
    #     os.remove(tmp_path)        # cleanup the original .avif
    #     tmp_path = png_path        # point to the converted file

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

    # ----------------------------------------------------------
    # 1⃣  Auto-wrap: first try 2 lines, then (optionally) 3 lines
    # ----------------------------------------------------------
    max_line_width = int(video_size[0] * 0.9) - 2 * padding
    words = text.split()

    def line_width(s: str) -> int:
        return draw_dummy.textbbox((0, 0), s, font=font)[2]

    def split_evenly(ws: list[str]) -> tuple[str, str]:
        """Return the 'best' two-line split of the words list."""
        best_split, best_score = None, None
        for i in range(1, len(ws)):
            left, right = " ".join(ws[:i]), " ".join(ws[i:])
            wl, wr = line_width(left), line_width(right)
            if max(wl, wr) <= max_line_width:                  # legal split
                score = abs(wl - wr)                           # balance metric
                if best_score is None or score < best_score:
                    best_split, best_score = i, score
        if best_split is None:                                 # fallback = middle
            best_split = len(ws) // 2
        return " ".join(ws[:best_split]), " ".join(ws[best_split:])

    # ---- first, try to reduce to ≤2 lines
    one_line_width = line_width(text)
    if one_line_width > max_line_width and len(words) > 1:
        line_a, line_b = split_evenly(words)
        text = f"{line_a}\n{line_b}"

        # ---- now, if either line is STILL too wide, try a third line
        # (we only split the widest of the two lines).
        lines = text.split("\n")
        widest_idx = max(range(len(lines)), key=lambda i: line_width(lines[i]))
        if (line_width(lines[widest_idx]) > max_line_width          # still too wide
                and len(lines) < 3                                  # keep max=3
                and len(lines[widest_idx].split()) > 1):            # >1 word
            sub_a, sub_b = split_evenly(lines[widest_idx].split())
            # insert the two new lines in place of the oversize one
            lines = lines[:widest_idx] + [sub_a, sub_b] + lines[widest_idx+1:]
            text = "\n".join(lines)
    # ----------------------------------------------------------

    
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

            lift = int(0.20 * H)                      # 25 % of full height
            y = max(margin, H - clip_h - lift)        # never go above the top margin
            return ((W - clip_w) // 2, y)
            

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


def build_kb_preset(preset_id: int, w: int, Z: float) -> list[tuple[float, float]]:
    max_off = (w / Z - w) / -2.0      # pan as far as the edge of the zoomed view
    half    = 0.5 * max_off           # a bit gentler

    presets = {
        1: [(1.0, 0.0), (Z,   0.0)],                  # centre zoom-in
        2: [(Z,   0.0), (1.0, 0.0)],                  # centre zoom-out
        3: [(1.0, -half), (1.0, half)],               # pan left → right
        4: [(1.0,  half), (1.0,-half)],               # pan right → left
        5: [(1.0, -half), (Z,   -half)],              # zoom-in on left side
        6: [(1.0,  half), (Z,    half)],              # zoom-in on right side
    }
    if preset_id not in presets:
        raise ValueError("Ken Burns preset must be 0 or 1–6.")
    return presets[preset_id]

def smoothstep(u: float) -> float:
    """Smoothstep easing function for u in [0,1]."""
    return 3*u**2 - 2*u**3

def apply_ken_burns_effect(
    image_clip: mpe.ImageClip,
    duration: float,
    start_hold: float = 0.5,
    intensity_min: float = 1.2,
    intensity_max: float = 1.4,
    preset: int = 0
) -> mpe.VideoClip:
    """
    A Ken Burns-style effect that chooses a sequence of zoom/pan states
    and smooth-transitions between them over 'duration' seconds.

    Main features:
      - If duration < 6, we randomly pick either a 2-phase or 1-phase
        sequence (so short clips can have just one movement).
      - For longer durations, we choose from multiple multi-phase sequences.
      - We avoid:
         * Zooming out directly from full view,
         * Immediate pan direction reversals (e.g., right->left),
         * Immediate zoom in->out or out->in,
         * No-op sequences with no actual movement.
      - We clamp the first phase (if it's long) to 'start_hold' seconds,
        ensuring the effect starts earlier rather than lingering at full view.
      - We respect exactly the user-specified `duration` by scaling the
        internal transition boundaries, so the final clip is `duration` long.
    """

    # ----------------------
    # 1) Basic Setup
    # ----------------------
    phase_min = 2        # Minimum duration for each phase
    w, h = image_clip.size

    # Random zoom factor for "interesting" states
    Z = random.uniform(intensity_min, intensity_max)

    # Calculate maximum horizontal offset for the chosen zoom
    new_w_at_Z = w / Z
    max_possible_offset = (w - new_w_at_Z) / 2.0

    # Offsets for slides when zoomed in
    slide_offset_min = 0.5 * max_possible_offset
    slide_offset_max = max_possible_offset

    # Smaller offset for pre-slide at full view
    pre_max = max(5, 0.05 * w)


    if preset != 0:                                    # 1–6
        
        key_states = build_kb_preset(preset, w, Z)

    else:

        # ----------------------
        # 2) Pick a sequence of key states (zoom, offset)
        # ----------------------
        #if duration < 6:
        if True:
            # Sequences for short videos
            short_sequences = []
            # Two-phase sequences (3 key states, i.e. 2 transitions)
            #ss1 = [(1.0, 0.0), (Z, 0.0), (1.0, 0.0)]
            ss1 = [(1.0, 0.0), (Z, 0.0)]

            offset = random.uniform(slide_offset_min, slide_offset_max)
            if random.choice([True, False]):
                offset = -offset
            #ss2 = [(1.0, 0.0), (Z, offset), (1.0, 0.0)]

            ss2 = [(1.0, 0.0), (Z, offset)]
            short_sequences.append(ss1)
            short_sequences.append(ss2)
            # Single-phase sequence (2 key states, i.e. 1 transition)
            offset = random.uniform(slide_offset_min, slide_offset_max)
            if random.choice([True, False]):
                offset = -offset
            ss_single = [(1.0, 0.0), (Z, offset)]
            short_sequences.append(ss_single)
            key_states = random.choice(short_sequences)
        else:
            # Sequences for longer videos
            long_sequences = []
            # Example A
            offset = random.uniform(slide_offset_min, slide_offset_max)
            if random.choice([True, False]):
                offset = -offset
            seqA = [(1.0, 0.0), (Z, 0.0), (Z, offset), (1.0, 0.0)]
            long_sequences.append(seqA)
            # Example B
            pre_offset = random.uniform(0, pre_max) * random.choice([-1, 1])
            seqB = [(1.0, 0.0), (1.0, pre_offset), (Z, pre_offset), (1.0, 0.0)]
            long_sequences.append(seqB)
            # Example C
            pre_offset = random.uniform(0, pre_max) * random.choice([-1, 1])
            offset = random.uniform(slide_offset_min, slide_offset_max)
            if random.choice([True, False]):
                offset = -offset
            seqC = [(1.0, 0.0), (1.0, pre_offset), (Z, pre_offset),
                    (Z, offset), (1.0, 0.0)]
            long_sequences.append(seqC)
            # Example D
            offset1 = random.uniform(slide_offset_min, slide_offset_max) * random.choice([-1, 1])
            offset2 = random.uniform(slide_offset_min, slide_offset_max) * random.choice([-1, 1])
            seqD = [(1.0, 0.0), (Z, 0.0), (Z, offset1), (Z, offset2), (1.0, 0.0)]
            long_sequences.append(seqD)
            # Example E
            pre_offset = random.uniform(0, pre_max) * random.choice([-1, 1])
            offset = random.uniform(slide_offset_min, slide_offset_max)
            if random.choice([True, False]):
                offset = -offset
            seqE = [(1.0, 0.0), (1.0, pre_offset), (Z, pre_offset),
                    (Z, offset), (Z, pre_offset), (1.0, 0.0)]
            long_sequences.append(seqE)
            key_states = random.choice(long_sequences)

    # ----------------------
    # 3) Safe transitions
    #    (3.1) Avoid zoom-out from full view
    # ----------------------
    for i in range(1, len(key_states)):
        nz, noff = key_states[i]
        if nz == 1.0 or noff != 0:
            pz, poff = key_states[i-1]
            if pz == 1.0:
                key_states[i-1] = (Z, poff)

    # ----------------------
    # 3.2) Avoid immediate pan direction reversal (R->L or L->R)
    # Instead of zeroing the second offset, reduce it.
    # ----------------------
    for i in range(1, len(key_states)):
        prev_offset = key_states[i-1][1]
        cur_offset  = key_states[i][1]
        if prev_offset != 0 and cur_offset != 0:
            if (prev_offset * cur_offset) < 0:
                key_states[i] = (key_states[i][0], cur_offset * 0.5)

    # ----------------------
    # 4) Avoid immediate zoom in->out or out->in by removing the middle state.
    # ----------------------
    def zoom_direction(z1, z2):
        if z2 > z1:
            return "IN"
        elif z2 < z1:
            return "OUT"
        else:
            return "NONE"

    i = 0
    while i < len(key_states) - 2:
        z1, _ = key_states[i]
        z2, _ = key_states[i+1]
        z3, _ = key_states[i+2]
        d1 = zoom_direction(z1, z2)
        d2 = zoom_direction(z2, z3)
        if (d1 == "IN" and d2 == "OUT") or (d1 == "OUT" and d2 == "IN"):
            key_states.pop(i+1)  # remove the middle state
            if i > 0:
                i -= 1
        else:
            i += 1

    # ----------------------
    # 5) Ensure there's at least some difference in states
    #    (avoid the scenario of two consecutive states being identical)
    # ----------------------
    def states_have_movement(kstates):
        """Check if there's any difference in zoom or offset across these states."""
        for idx in range(len(kstates) - 1):
            z1, off1 = kstates[idx]
            z2, off2 = kstates[idx+1]
            if abs(z1 - z2) > 1e-3 or abs(off1 - off2) > 1e-2:
                return True
        return False

    if len(key_states) == 1:
        z, off = key_states[0]
        new_z = z + 0.05 if z == 1.0 else z + 0.02
        key_states = [key_states[0], (new_z, off)]
    elif not states_have_movement(key_states):
        z_last, off_last = key_states[-1]
        small_zoom = z_last + 0.05 if abs(z_last - 1.0) < 0.01 else z_last + 0.02
        key_states[-1] = (small_zoom, off_last)

    # --- NEW STEP: Remove duplicate consecutive states ---
    filtered_key_states = [key_states[0]]
    for state in key_states[1:]:
        prev = filtered_key_states[-1]
        # If both zoom and offset are nearly identical, skip adding this duplicate state.
        if abs(state[0] - prev[0]) < 1e-3 and abs(state[1] - prev[1]) < 1e-2:
            continue
        filtered_key_states.append(state)
    key_states = filtered_key_states

    # --- NEW STEP: Ensure minimum zoom difference when offsets are zero ---
    # If both consecutive states have zero offset and the zoom difference is less than 0.2,
    # force the difference to be exactly 0.2.
    for i in range(1, len(key_states)):
        prev_zoom, prev_off = key_states[i-1]
        curr_zoom, curr_off = key_states[i]
        if abs(prev_off) < 1e-2 and abs(curr_off) < 1e-2:  # both offsets zero
            if curr_zoom - prev_zoom < 0.2:
                key_states[i] = (prev_zoom + 0.2, curr_off)

    print("Debug: Final key states after duplicate removal and zoom adjustment:")
    for idx, state in enumerate(key_states):
        print(f"  State {idx}: Zoom = {state[0]:.3f}, Offset = {state[1]:.3f}")

    # ----------------------
    # 6) Compute phase durations
    # ----------------------
    n_phases = len(key_states) - 1
    if n_phases <= 0:
        key_states = [(1.0, 0.0), (1.05, 0.0)]
        n_phases = 1

    required_min = phase_min * n_phases
    if duration < required_min:
        phase_durations = [duration / n_phases] * n_phases
        total_duration = duration
    else:
        extra = duration - required_min
        partition = np.random.dirichlet(np.ones(n_phases)) * extra
        phase_durations = [phase_min + part for part in partition]
        total_duration = sum(phase_durations)

    # ----------------------
    # 7) Optionally reduce the first phase if it's long (start_hold)
    # ----------------------
    if n_phases > 1 and phase_durations[0] > start_hold:
        diff = phase_durations[0] - start_hold
        phase_durations[0] = start_hold
        for i in range(1, n_phases):
            phase_durations[i] += diff / (n_phases - 1)
        total_duration = sum(phase_durations)

    # ----------------------
    # 8) Build boundaries, then scale them to match exactly 'duration'
    # ----------------------
    boundaries = [0]
    for d in phase_durations:
        boundaries.append(boundaries[-1] + d)

    if abs(total_duration - duration) > 1e-6:
        scale = duration / total_duration
        for i in range(1, len(boundaries)):
            boundaries[i] *= scale
        total_duration = duration

    print("Debug: Boundaries after scaling:", [f"{b:.2f}" for b in boundaries])
    print("Debug: Total duration =", total_duration)

    # ----------------------
    # 9) Frame function for the Ken Burns effect
    # ----------------------
    def ken_burns(get_frame, t):
        phase_index = 0
        for j in range(n_phases):
            if boundaries[j] <= t < boundaries[j+1]:
                phase_index = j
                break
        else:
            phase_index = n_phases - 1

        t_phase = t - boundaries[phase_index]
        phase_length = boundaries[phase_index+1] - boundaries[phase_index]
        u = max(0, min(t_phase / phase_length, 1))
        u_eased = smoothstep(u)

        start_zoom, start_offset = key_states[phase_index]
        end_zoom, end_offset = key_states[phase_index+1]
        current_zoom = start_zoom + (end_zoom - start_zoom) * u_eased
        current_offset = start_offset + (end_offset - start_offset) * u_eased

        crop_w = w / current_zoom
        crop_h = h / current_zoom
        center_x = w / 2 + current_offset
        center_y = h / 2

        center_x = max(crop_w/2, min(center_x, w - crop_w/2))
        center_y = max(crop_h/2, min(center_y, h - crop_h/2))

        frame = get_frame(t)

        if frame.dtype != np.uint8:
            frame = (frame * 255).clip(0, 255).astype(np.uint8)

        patch = cv2.getRectSubPix(
            frame,
            (int(round(crop_w)), int(round(crop_h))),
            (center_x, center_y)
        )
        resized = cv2.resize(patch, (w, h), interpolation=cv2.INTER_LINEAR)
        return resized

    image_clip = image_clip.set_duration(duration)
    return image_clip.fl(ken_burns, apply_to=['mask','video'])






def sync_videos_to_song(video_info: list, song_file: str, do_trim: bool, effect_hold: float, intensity_min: float, intensity_max: float, add_sub: bool, add_audio: bool, output_file: str,
                        loop_count: int = 1, aspect_ratio: str = "16:9",
                        target_resolution: tuple = None, ideal_dur: float = 0.0):
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

    

    first_video_position = video_info[0].get("position")

    if first_video_position == "random":

        shuffle_pos = True
        print("random positioning")

    else:

        shuffle_pos = False

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
            ken_burns         = video.get("ken_burns", True)
            fadein_duration   = video.get("fadein_duration", 0.5)
            kb_preset         = video.get("ken_burns_preset", 0)

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
            ken_burns         = True
            hex_color         = "#000000"
            kb_preset         = 0
           
        

        try:

            local_path = download_file(url)

            local_path = normalise_exif_orientation(local_path)

            downloaded_inputs.append((local_path, caption, ken_burns, kb_preset))



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

   
    if ideal_dur == 0:                       # no user preference → use default window
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

    previous_caption = None          # remember last one

    for idx, (seg_start, seg_end) in enumerate(segments):
        if idx >= allowed_segments:
            break
        target_duration = seg_end - seg_start
        
        local_path, caption, ken_burns, kb_preset = all_inputs[idx]

        print(f"Processing '{local_path}' for segment {idx+1} with target duration {target_duration:.2f}s")
        if local_path.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif', '.tiff', '.tif', '.svg', '.heif', '.heic', '.ico', '.jfif', '.pjpeg', '.pjp', '.avif')):
            base_clip = mpe.ImageClip(local_path).set_duration(target_duration)

            if ken_burns:
                clip = apply_ken_burns_effect(base_clip, target_duration, effect_hold, intensity_min, intensity_max, kb_preset)
            else:
                clip = base_clip
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

        if caption and add_sub:

            #pos = random.choice([ "top-left", "top-center", "top-right", "bottom-left", "bottom-center", "bottom-right", "middle" ])
            
            this_fadein = 0 if caption == previous_caption else fadein_duration

            if shuffle_pos:
                position = random.choice(POSSIBLE_POSITIONS)


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
                fadein_duration=this_fadein
            )

            clip = mpe.CompositeVideoClip([clip, caption_clip])

        video_clips.append(clip)
        previous_caption = caption
    
    final_video = mpe.concatenate_videoclips(video_clips, method="compose")
    final_duration = final_video.duration

    song_dur = song_audio.duration

    print(f"Final video duration: {final_duration:.6f}")
    print(f"Song audio duration:  {song_dur:.6f}")
    print(f"Difference (video - song): {final_duration - song_dur:.6f}")

    if final_duration <= song_dur:
        # If the final video is shorter or equal to the song, we just subclip.
        final_audio = song_audio.subclip(0, final_duration)
    else:
        # If the final video is longer, we loop the audio until it matches.
        looped_audio = audio_loop(song_audio, duration=final_duration)
        final_audio = looped_audio.subclip(0, final_duration)

    if add_audio:
        
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
