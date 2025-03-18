import cv2
import numpy as np
import random
from augmentation_strategy import AugmentationStrategy

class BackgroundChangeAugmentation(AugmentationStrategy):

    def graysacale_image(self, img):  
        channels = img.shape[2] if len(img.shape) > 2 else 1
       
        if channels == 1:
            img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif channels == 3:
            img_color = img
        elif channels == 4:
            img_color = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img_color

    def create_contour_mask(self, image):
        """Create mask following the actual object contours"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold to separate object from background
            _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                raise ValueError("No contours found in image")
            
            # Get largest contour (assuming it's the main object)
            main_contour = max(contours, key=cv2.contourArea)
            
            # Create mask from contour
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [main_contour], -1, (255), thickness=cv2.FILLED)
            
            # Convert to 3-channel mask
            mask = mask / 255.0
            mask = np.stack([mask] * 3, axis=-1)
            
            return mask
            
        except Exception as e:
            raise ValueError(f"Failed to create contour mask: {str(e)}")
        
    def apply(self, image, config=None):
        try:
            if image is None or image.size == 0:
                raise ValueError("Invalid input image")

            # Create background options
            bg_colors = {
                "black": np.zeros_like(image),
                "white": np.full_like(image, 255),
                "grayscale": self.graysacale_image(image)
            }
            
            # Get contour-based mask
            object_mask = self.create_contour_mask(image)
            
            # Select random background
            new_background = random.choice(list(bg_colors.values()))
            
            # Combine: keep only the object pixels, change everything else
            result = image * object_mask + new_background * (1 - object_mask)
            
            if result is None:
                raise ValueError("Failed to create augmented image")
                
            return result.astype(np.uint8)
            
        except Exception as e:
            raise ValueError(f"Background change failed: {str(e)}")