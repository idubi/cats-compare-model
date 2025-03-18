from abc import ABC, abstractmethod

class AugmentationStrategy(ABC):
    """Abstract base class for all augmentation strategies."""

    @abstractmethod
    def apply(self, image, config=None):
        """
        Args:
            image: The image array
            config: Dictionary containing:
                - image_path: path to image file
                - detection_box: bbox coordinates
                - other parameters
        """
        pass
