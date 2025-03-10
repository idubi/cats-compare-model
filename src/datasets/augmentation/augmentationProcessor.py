import os
import cv2
import json
import random
import numpy as np
import albumentations as A
from tqdm import tqdm
from PIL import Image

# === CONFIGURATION ===
BASE_DIR = "./DATASETS/BLUEPRINT_DS"

CATEGORIES = ["CAT", "DOG"]
IMAGE_SIZE = (224, 224)
YOLO_CONFIDENCE_THRESHOLD = 0.5  # Ignore images below this confidence

# Augmentation probability distribution (Total = 100%)
AUGMENTATION_PROBABILITIES = {
    "rotate": 15,  # 15%
    "flip": 10,  # 10%
    "grayscale": 5,  # 5%
    "sepia": 3,  # 3%
    "noise": 7,  # 7%
    "sharpen": 10,  # 10%
    "crop": 20,  # 20%
    "background_change": 20,  # 20%
    "negative": 10  # 10%
}

# Background replacement options
BACKGROUND_TEXTURES = ["grass", "wall", "sand", "wood", "black", "white", "grayscale", "sepia"]

# Augmentations per image
AUGMENTED_IMAGES_PER_ORIGINAL = 5
PAIRS_PER_CATEGORY = 5000
DEBUG_MODE = False  # Set to True to visualize 5% of augmentations

# === CLEAN UP BEFORE STARTING ===
def cleanup_augmentation():
    print("🗑️ Deleting existing augmented dataset...")
    for category in CATEGORIES:
        aug_dir = os.path.join(BASE_DIR, category, "AUGMENTED")
        if os.path.exists(aug_dir):
            for root, _, files in os.walk(aug_dir):
                for file in files:
                    os.remove(os.path.join(root, file))
            os.makedirs(os.path.join(aug_dir, "images"), exist_ok=True)
            os.makedirs(os.path.join(aug_dir, "labels"), exist_ok=True)

# === AUGMENTATION FUNCTIONS ===
def apply_augmentation(image, aug_type):
    """Applies a specific augmentation to an image."""
    if aug_type == "rotate":
        return A.Rotate(limit=15, p=1.0)(image=image)["image"]
    elif aug_type == "flip":
        return cv2.flip(image, 1)  # Horizontal flip
    elif aug_type == "grayscale":
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif aug_type == "sepia":
        sepia_filter = np.array([[0.272, 0.534, 0.131],
                                 [0.349, 0.686, 0.168],
                                 [0.393, 0.769, 0.189]])
        return cv2.transform(image, sepia_filter)
    elif aug_type == "noise":
        noise = np.random.normal(0, 0.02, image.shape).astype(np.uint8)
        return cv2.add(image, noise)
    elif aug_type == "sharpen":
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        return cv2.filter2D(image, -1, kernel)
    elif aug_type == "crop":
        scale = random.uniform(0.8, 1.0)
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        y_offset, x_offset = random.randint(0, h - new_h), random.randint(0, w - new_w)
        return image[y_offset:y_offset + new_h, x_offset:x_offset + new_w]
    elif aug_type == "negative":
        return cv2.bitwise_not(image)
    elif aug_type == "background_change":
        bg_color = random.choice(BACKGROUND_TEXTURES)
        if bg_color == "black":
            return np.zeros_like(image)
        elif bg_color == "white":
            return np.full_like(image, 255)
        elif bg_color == "grayscale":
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        elif bg_color == "sepia":
            return apply_augmentation(image, "sepia")
        return image  # No change

# === GENERATE AUGMENTATIONS ===
def generate_augmentations():
    print("🎨 Generating augmented images...")
    augmentation_log = {}

    for category in CATEGORIES:
        original_dir = os.path.join(BASE_DIR, category, "ORIGINAL", "images")
        augmented_dir = os.path.join(BASE_DIR, category, "AUGMENTED", "images")

        image_files = os.listdir(original_dir)
        for image_name in tqdm(image_files):
            image_path = os.path.join(original_dir, image_name)
            image = cv2.imread(image_path)

            # Skip low-confidence images
            if random.random() > YOLO_CONFIDENCE_THRESHOLD:
                continue

            for _ in range(AUGMENTED_IMAGES_PER_ORIGINAL):
                aug_image = image.copy()
                aug_id = f"{random.randint(10000, 99999)}.jpg"

                # Apply random augmentations based on probability
                applied_augmentations = []
                for aug, prob in AUGMENTATION_PROBABILITIES.items():
                    if random.randint(1, 100) <= prob:
                        aug_image = apply_augmentation(aug_image, aug)
                        applied_augmentations.append(aug)

                # Save augmented image
                aug_save_path = os.path.join(augmented_dir, aug_id)
                cv2.imwrite(aug_save_path, aug_image)

                # Log applied augmentations
                augmentation_log[aug_id] = applied_augmentations

    # Save augmentation log
    with open(os.path.join(BASE_DIR, "augmentations.json"), "w") as f:
        json.dump(augmentation_log, f, indent=4)

# === GENERATE PAIRS FOR TRAINING ===
def generate_pairs():
    print("🔗 Generating pairs for Siamese learning...")
    pairs = {}

    for category in CATEGORIES:
        images = os.listdir(os.path.join(BASE_DIR, category, "AUGMENTED", "images"))
        pet_groups = {}

        # Group images by pet ID
        for img in images:
            pet_id = img.split("_")[0]  # Extract pet ID
            if pet_id not in pet_groups:
                pet_groups[pet_id] = []
            pet_groups[pet_id].append(img)

        pairs[category] = []
        pet_ids = list(pet_groups.keys())

        for _ in range(PAIRS_PER_CATEGORY):
            # Positive pair (same pet)
            pet_id = random.choice(pet_ids)
            if len(pet_groups[pet_id]) > 1:
                img1, img2 = random.sample(pet_groups[pet_id], 2)
                pairs[category].append({"img1": img1, "img2": img2, "label": 1})

            # Negative pair (different pets)
            pet1, pet2 = random.sample(pet_ids, 2)
            pairs[category].append({"img1": random.choice(pet_groups[pet1]), "img2": random.choice(pet_groups[pet2]), "label": 0})

    with open(os.path.join(BASE_DIR, "pairs.json"), "w") as f:
        json.dump(pairs, f, indent=4)

# === RUN THE SYSTEM ===
cleanup_augmentation()
generate_augmentations()
generate_pairs()
print("✅ Augmentation complete!")
