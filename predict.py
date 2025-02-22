import os
import tempfile
import json
from cog import BasePredictor, Input, Path
from main import sync_videos_to_song, download_file

class Predictor(BasePredictor):
    def setup(self):
        # Any expensive setup can be done here.
        pass

    def predict(
        self,
        video_info: str = Input(
            description="A JSON string representing a list of objects with keys 'url' and optional 'caption'.",
            default='[{"url": "https://example.com/video.mp4", "caption": "Sample Caption"}]'
        ),
        song_url: str = Input(
            description="URL to the song (audio file).",
            default="https://example.com/song.mp3"
        ),
        loop_count: int = Input(
            description="How many times to loop the input list.",
            default=1
        ),
        aspect_ratio: str = Input(
            description="Output aspect ratio (e.g. '16:9' or '9:16').",
            default="16:9"
        ),
        do_trim: bool = Input(
            description="Trim long song", default=False
        ),
        target_resolution: str = Input(
            description="Output resolution in WIDTHxHEIGHT format (e.g. '1920x1080').",
            default="1920x1080"
        )
    ) -> Path:
        # Parse video_info JSON string into a list of dictionaries.
        try:
            video_info_parsed = json.loads(video_info)
        except Exception as e:
            raise ValueError("video_info must be a valid JSON string.") from e

        # Parse target resolution string.
        try:
            width, height = map(int, target_resolution.lower().split("x"))
        except Exception as e:
            raise ValueError("target_resolution must be in WIDTHxHEIGHT format, e.g. '1920x1080'.") from e
        target_resolution_tuple = (width, height)
        
        # Download the song file.
        song_file_path = download_file(song_url)
        
        # Create a temporary output file.
        output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        
        # Run the media-processing function.
        sync_videos_to_song(
            video_info_parsed,
            song_file_path,
            do_trim,
            output_file,
            loop_count=loop_count,
            aspect_ratio=aspect_ratio,
            target_resolution=target_resolution_tuple
        )
        
        # Clean up the downloaded song file.
        os.remove(song_file_path)
        
        # Return the path to the output video.
        return Path(output_file)
