
import random
# import sys 
import os
import cv2
from augmentations.rotate import RotateAugmentation
from augmentations.flip import FlipAugmentation
from augmentations.grayscale import GrayscaleAugmentation
from augmentations.background_change import BackgroundChangeAugmentation
from augmentations.crop import CropAugmentation

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))


# sys.path.append(project_root)

# Now you can import using the full path from the project root
# from src.services.logger_service import LoggerService

# Get the current module name without extension
# module_name = os.path.splitext(os.path.basename(__file__))[0]  # This will give 'BluePrintUtils'

# Initialize logger with module name
# logger = LoggerService(module_name)

class AugmentationExecutor:
    """Executes multiple augmentations based on configuration and probabilities."""

    def __init__(self, config_manager):
        self.config = config_manager
        self.augmentations = self._load_augmentations()

    def get_bbox_from_label(self,image_path):
        if not image_path:
            raise Exception("image_path is empty")
        image_name = os.path.basename(image_path)
        # Get bbox from corresponding label
        label_path = image_path.replace('images', 'labels').replace('.jpg', '.txt')

        img = cv2.imread(image_path)
        img_height, img_width = img.shape[:2]
        
        try:
            with open(label_path, 'r') as f:
                # YOLO format: class x_center y_center width height
                class_id, x_center, y_center, width, height = map(float, f.read().strip().split())
                
                # Convert YOLO format (normalized) to pixel coordinates
                x1 = int((x_center - width/2) * img_width)
                y1 = int((y_center - height/2) * img_height)
                x2 = int((x_center + width/2) * img_width)
                y2 = int((y_center + height/2) * img_height)
                
                return [x1, y1, x2, y2]
        except Exception as e:
            raise Exception(f"Error reading label file {label_path}: {str(e)}")
        
        
     
        
    
    def _load_augmentations(self):
        augmentation_map = {
            "rotate"           : RotateAugmentation()           ,
            "flip"             : FlipAugmentation()             ,
            "grayscale"        : GrayscaleAugmentation()        ,
            "background_change": BackgroundChangeAugmentation() ,
            "crop"             : CropAugmentation()            
        }
        enabled_augmentations = {}
        for aug_name, aug_instance in augmentation_map.items():
            if self.config.is_augmentation_enabled(aug_name):
                enabled_augmentations[aug_name] = {
                    "instance": aug_instance,
                    "probability": self.config.get_augmentation_probability(aug_name)
                }
        return enabled_augmentations

    def apply_augmentations(self, image,image_path=None):
        augmented_image = image.copy()
        applied_augmentations = []

        for aug_name, aug in self.augmentations.items():
            if random.uniform(0, 100) <= aug["probability"]:
                if aug_name == 'background_change' :
                    bbox = self.get_bbox_from_label(image_path)
                    config = {
                                    'detection_box': bbox,
                                    'params': {}
                              }
                    augmented_image = aug["instance"].apply(augmented_image,config)
                else :
                    augmented_image = aug["instance"].apply(augmented_image)
                applied_augmentations.append(aug_name)
        return augmented_image, applied_augmentations
