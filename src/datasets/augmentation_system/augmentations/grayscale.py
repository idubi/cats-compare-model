import cv2
from augmentation_strategy import AugmentationStrategy

class GrayscaleAugmentation(AugmentationStrategy):
    def apply(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
