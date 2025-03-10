import cv2
import numpy as np
import random
from abc import ABC, abstractmethod

class AugmentationStrategy(ABC):
    """Abstract base class for all augmentation strategies."""

    @abstractmethod
    def apply(self, image):
        """Applies the augmentation to the image."""
        pass

class RotateAugmentation(AugmentationStrategy):
    """Rotates an image by a random degree within a given range."""
    
    def __init__(self, angle=15):
        self.angle = angle

    def apply(self, image):
        height, width = image.shape[:2]
        angle = random.uniform(-self.angle, self.angle)
        rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        return cv2.warpAffine(image, rotation_matrix, (width, height))

class FlipAugmentation(AugmentationStrategy):
    """Flips an image horizontally or vertically."""
    
    def apply(self, image):
        flip_type = random.choice([-1, 0, 1])  # Horizontal, Vertical, Both
        return cv2.flip(image, flip_type)

class GrayscaleAugmentation(AugmentationStrategy):
    """Converts an image to grayscale."""
    
    def apply(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

class BackgroundChangeAugmentation(AugmentationStrategy):
    """Replaces the background of an image with a random solid color or texture."""
    
    def apply(self, image):
        bg_colors = {
            "black": np.zeros_like(image),
            "white": np.full_like(image, 255),
            "grayscale": cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
            "sepia": np.array([[0.272, 0.534, 0.131],
                               [0.349, 0.686, 0.168],
                               [0.393, 0.769, 0.189]])
        }
        selected_bg = random.choice(list(bg_colors.keys()))
        if selected_bg == "sepia":
            return cv2.transform(image, bg_colors[selected_bg])
        return bg_colors[selected_bg]

class NoiseAugmentation(AugmentationStrategy):
    """Adds Gaussian noise to an image."""
    
    def apply(self, image):
        noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
        return cv2.add(image, noise)

class CropAugmentation(AugmentationStrategy):
    """Crops an image randomly within a given range."""
    
    def apply(self, image):
        h, w = image.shape[:2]
        scale = random.uniform(0.8, 1.0)
        new_h, new_w = int(h * scale), int(w * scale)
        y_offset, x_offset = random.randint(0, h - new_h), random.randint(0, w - new_w)
        return image[y_offset:y_offset + new_h, x_offset:x_offset + new_w]

class NegativeAugmentation(AugmentationStrategy):
    """Inverts the colors of an image (negative effect)."""
    
    def apply(self, image):
        return cv2.bitwise_not(image)