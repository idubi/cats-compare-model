from augmentation_strategy import AugmentationStrategy
import random
import cv2

class CropAugmentation(AugmentationStrategy):
    def __init__(self):
        super().__init__()
        self.name = "crop"

    def apply(self, image, config):
        """
        Apply crop augmentation ensuring detected object remains in frame
        Args:
            image: Input image
            config: Configuration dictionary containing:
                   - scale: [min_scale, max_scale]
                   - detection_box: [x1, y1, x2, y2] from YOLO
        """
        try:
            scale = config.get('params', {}).get('scale', [0.7, 0.9])
            detection_box = config.get('detection_box')
            
            if detection_box is None:
                return None  # Can't crop safely without knowing object location
            
            img_height, img_width = image.shape[:2]
            scale_factor = random.uniform(scale[0], scale[1])
            
            # Get object boundaries
            obj_x1, obj_y1, obj_x2, obj_y2 = detection_box
            
            # Calculate crop dimensions
            crop_width = int(img_width * scale_factor)
            crop_height = int(img_height * scale_factor)
            
            # Calculate valid ranges for crop start position
            # Ensure object will be inside crop
            x_start_min = max(0, obj_x2 - crop_width)  # Don't cut off right side
            x_start_max = min(obj_x1, img_width - crop_width)  # Don't cut off left side
            y_start_min = max(0, obj_y2 - crop_height)  # Don't cut off bottom
            y_start_max = min(obj_y1, img_height - crop_height)  # Don't cut off top
            
            # If ranges are invalid, adjust crop size
            if x_start_min > x_start_max or y_start_min > y_start_max:
                return None
            
            # Random position within valid ranges
            x_start = random.randint(int(x_start_min), int(x_start_max))
            y_start = random.randint(int(y_start_min), int(y_start_max))
            
            # Perform crop
            cropped = image[y_start:y_start + crop_height, 
                          x_start:x_start + crop_width]
            
            if cropped.size == 0:
                return None

            return cropped

        except Exception as e:
            return None