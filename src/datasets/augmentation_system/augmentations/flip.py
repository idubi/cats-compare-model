import cv2
import random
from augmentation_strategy import AugmentationStrategy

class FlipAugmentation(AugmentationStrategy):
    def apply(self, image):
        return cv2.flip(image, 1)
