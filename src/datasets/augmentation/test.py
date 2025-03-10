 import ConfigManager from ConfigManager
 


# Initialize the ConfigManager (Loads settings from JSON)
config_manager = ConfigManager("config.json")

# Initialize the AugmentationProcessor
augmentation_processor = AugmentationProcessor(config_manager)

# Load an image
image = cv2.imread("path/to/cat_image.jpg")

# Apply augmentations
augmented_image, applied_augmentations = augmentation_processor.apply_augmentations(image)

# Save augmented image
cv2.imwrite("augmented_image.jpg", augmented_image)

# Print applied augmentations
print(f"Applied Augmentations: {applied_augmentations}")
