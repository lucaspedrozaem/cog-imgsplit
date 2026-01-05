# predict.py
import cv2
import numpy as np
import os
import tempfile
import shutil
import requests
from typing import List, Optional
from urllib.parse import urlparse
from cog import BasePredictor, Input, Path

class Predictor(BasePredictor):
    def setup(self):
        """Load necessary libraries or models (none needed for standard OpenCV)"""
        pass

    def download_image(self, url: str) -> str:
        """Helper to download image from URL to a temp file."""
        try:
            resp = requests.get(url, stream=True)
            resp.raise_for_status()
            
            # Create a temp file
            # We try to keep the extension from the url if possible, default to .jpg
            parsed = urlparse(url)
            ext = os.path.splitext(parsed.path)[1]
            if not ext:
                ext = ".jpg"
                
            tmp_file = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
            with open(tmp_file.name, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return tmp_file.name
        except Exception as e:
            raise ValueError(f"Failed to download image from {url}: {e}")

    def predict(
        self,
        image_url: str = Input(
            description="Public URL to the input image containing the photo grid."
        ),
        threshold_value: int = Input(
            description="Brightness threshold (0-255). Pixels brighter than this count as background.",
            default=230,
            ge=0, le=255
        ),
        min_height_ratio: float = Input(
            description="Minimum height of a detected photo relative to total image height (0.2 = 20%).",
            default=0.20,
            ge=0.01, le=1.0
        ),
        padding: int = Input(
            description="Padding pixels to add back to the cropped area.",
            default=6
        )
    ) -> List[Path]:
        """
        Download image from URL, detect vertical photos, and return cropped paths.
        """
        
        # 1. Download Image
        if not image_url or image_url == "null":
             raise ValueError("Please provide a valid image_url.")
             
        local_img_path = self.download_image(image_url)
        
        try:
            # 2. Load Image with OpenCV
            img = cv2.imread(local_img_path)
            if img is None:
                raise ValueError(f"CV2 could not read image from {image_url}")

            img_h, img_w = img.shape[:2]

            # 3. Convert to Grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 4. Inverse Thresholding
            # Background (>threshold) becomes Black. Photos (<threshold) become White.
            _, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)

            # 5. Morphological Clean-up
            kernel_fill = np.ones((3, 3), np.uint8)
            thresh = cv2.dilate(thresh, kernel_fill, iterations=2)
            
            kernel_sep = np.ones((3, 3), np.uint8)
            thresh = cv2.erode(thresh, kernel_sep, iterations=4)

            # 6. Find Contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            valid_boxes = []

            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                
                # Filter A: Aspect Ratio (Must be Vertical: Width < Height)
                if w >= h:
                    continue

                # Filter B: Minimum Height
                if h < (img_h * min_height_ratio):
                    continue

                # Restore padding
                x_new = max(0, x - padding)
                y_new = max(0, y - padding)
                w_new = min(img_w - x_new, w + (padding * 2))
                h_new = min(img_h - y_new, h + (padding * 2))
                
                valid_boxes.append((x_new, y_new, w_new, h_new))

            print(f"Detected {len(valid_boxes)} vertical photos.")

            if not valid_boxes:
                print("No valid photos found based on current filters.")
                return []

            # 7. Sort Grid (Top-Left to Bottom-Right)
            valid_boxes.sort(key=lambda b: b[1]) # Sort by Y
            
            rows = []
            if valid_boxes:
                current_row = [valid_boxes[0]]
                row_tolerance = valid_boxes[0][3] // 2 
                
                for i in range(1, len(valid_boxes)):
                    box = valid_boxes[i]
                    prev_box = current_row[-1]
                    
                    if abs(box[1] - prev_box[1]) < row_tolerance:
                        current_row.append(box)
                    else:
                        current_row.sort(key=lambda b: b[0]) # Sort row by X
                        rows.append(current_row)
                        current_row = [box]
                
                current_row.sort(key=lambda b: b[0])
                rows.append(current_row)
            
            sorted_boxes = [box for row in rows for box in row]

            # 8. Crop and Save to Temp Directory
            output_paths = []
            output_dir = tempfile.mkdtemp()

            for i, (x, y, w, h) in enumerate(sorted_boxes):
                crop = img[y:y+h, x:x+w]
                
                out_filename = f"photo_{i+1:03d}.jpg"
                out_path = os.path.join(output_dir, out_filename)
                
                cv2.imwrite(out_path, crop)
                output_paths.append(Path(out_path))
                print(f"Processed: {out_filename}")

            return output_paths

        finally:
            # Clean up the downloaded input file
            if os.path.exists(local_img_path):
                os.remove(local_img_path)