import cv2
import random
from augmentation_strategy import AugmentationStrategy

class RotateAugmentation(AugmentationStrategy):
    def apply(self, image):
        angle = random.uniform(-15, 15)
        h, w = image.shape[:2]
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        return cv2.warpAffine(image, matrix, (w, h))
