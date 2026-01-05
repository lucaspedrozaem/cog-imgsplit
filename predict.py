# predict.py
import cv2
import numpy as np
import os
import tempfile
import shutil
from typing import List
from cog import BasePredictor, Input, Path

class Predictor(BasePredictor):
    def setup(self):
        """Load necessary libraries or models (none needed for standard OpenCV)"""
        pass

    def predict(
        self,
        image: Path = Input(description="Input image containing the photo grid"),
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
        Process the image to detect vertical photos on white background
        and return them as a list of cropped images.
        """
        
        # 1. Load Image
        # Convert Cog Path to string for cv2
        img_path = str(image)
        img = cv2.imread(img_path)
        
        if img is None:
            raise ValueError(f"Could not load image at {img_path}")

        img_h, img_w = img.shape[:2]

        # 2. Convert to Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 3. Inverse Thresholding
        # Background (>threshold) becomes Black. Photos (<threshold) become White.
        _, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)

        # 4. Morphological Clean-up
        kernel_fill = np.ones((3, 3), np.uint8)
        thresh = cv2.dilate(thresh, kernel_fill, iterations=2)
        
        kernel_sep = np.ones((3, 3), np.uint8)
        thresh = cv2.erode(thresh, kernel_sep, iterations=4)

        # 5. Find Contours
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

        # 6. Sort Grid (Top-Left to Bottom-Right)
        # Sort by Y first
        valid_boxes.sort(key=lambda b: b[1]) 
        
        rows = []
        if valid_boxes:
            current_row = [valid_boxes[0]]
            # Tolerance for what constitutes the "same row"
            row_tolerance = valid_boxes[0][3] // 2 
            
            for i in range(1, len(valid_boxes)):
                box = valid_boxes[i]
                prev_box = current_row[-1]
                
                if abs(box[1] - prev_box[1]) < row_tolerance:
                    current_row.append(box)
                else:
                    # Sort the finished row by X
                    current_row.sort(key=lambda b: b[0]) 
                    rows.append(current_row)
                    current_row = [box]
            
            # Append final row
            current_row.sort(key=lambda b: b[0])
            rows.append(current_row)
        
        sorted_boxes = [box for row in rows for box in row]

        # 7. Crop and Save to Temp Directory
        output_paths = []
        temp_dir = tempfile.mkdtemp()

        for i, (x, y, w, h) in enumerate(sorted_boxes):
            crop = img[y:y+h, x:x+w]
            
            out_filename = f"photo_{i+1:03d}.jpg"
            out_path = os.path.join(temp_dir, out_filename)
            
            cv2.imwrite(out_path, crop)
            output_paths.append(Path(out_path))
            print(f"Processed: {out_filename}")

        # Replicate automatically handles uploading these Paths and returning URLs
        return output_paths