import os
import shutil
import json
import random
import cv2
import torch
import wget
import zipfile
import kaggle
import numpy as np
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from ultralytics import YOLO
from torchvision import transforms

    # "huggingface": [
    #     "oxford-iiit-pet",
    #     "microsoft/cats_vs_dogs",
    #     "stanford-dogs"
    # ],

    # "kaggle": [
    #     "tongpython/cat-and-dog",
    #     # "msarb2/dogs-cats-images" 
    #     # "ashwingupta3012/cat-and-dog-images"
    # ],


# === CONFIGURATION ===
# YOLO_MODEL = "yolov8l.pt"  # Use the largest available YOLO version
YOLO_MODEL = "yolov8n.pt"
DATASETS = {
    "huggingface": [
        "microsoft/cats_vs_dogs"
    ],
    "kaggle": [
        "tongpython/cat-and-dog" 
    ] 
}

OUTPUT_DIR = "./PET_RECOGNITION/DATASETS/BLUEPRINT_DS"
CLASSES = ["CAT", "DOG"]
IMAGE_SIZE = (224, 224)

# === SETUP DIRECTORIES ===
for category in CLASSES:
    os.makedirs(os.path.join(OUTPUT_DIR, category, "ORIGINAL"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, category, "AUGMENTED"), exist_ok=True)

# === YOLO MODEL ===
yolo_model = YOLO(YOLO_MODEL)

# === FUNCTION TO DOWNLOAD FROM HUGGING FACE ===
def download_from_huggingface():
    print("📥 Downloading images from Hugging Face...")
    for dataset_name in DATASETS["huggingface"]:
        dataset = load_dataset(dataset_name, split="train")
        for idx, sample in tqdm(enumerate(dataset), total=len(dataset)):
            image = sample["image"]
            image_path = f"hf_{dataset_name.replace('/', '_')}_{idx}.jpg"
            process_image(image, image_path)

# === FUNCTION TO DOWNLOAD FROM KAGGLE ===
def download_from_kaggle():
    print("📥 Downloading images from Kaggle...")
    for dataset_name in DATASETS["kaggle"]:
        kaggle.api.dataset_download_files(dataset_name, path="kaggle_temp", unzip=True)
    
    # Process downloaded images
    for root, _, files in os.walk("kaggle_temp"):
        for file in files:
            image_path = os.path.join(root, file)
            process_image(image_path, file)

    shutil.rmtree("kaggle_temp")  # Clean up

# === FUNCTION TO DOWNLOAD FROM IMAGENET ===
def download_from_imagenet():
    print("📥 Downloading images from ImageNet...")
    for category, synset in DATASETS["imagenet"].items():
        url = f"http://www.image-net.org/api/text/imagenet.synset.geturls?wnid={synset}"
        urls = wget.download(url).split("\n")

        for idx, img_url in tqdm(enumerate(urls), total=len(urls)):
            try:
                image_path = f"imagenet_{category}_{idx}.jpg"
                wget.download(img_url, image_path)
                process_image(image_path, image_path)
            except:
                continue  # Skip broken images

# === FUNCTION TO PROCESS IMAGE (YOLO CLASSIFICATION) ===
def process_image(image, image_name):
    """
    Runs YOLO on the image, classifies it as CAT or DOG, and saves it to BLUEPRINT_DS.
    Also, saves YOLO labels in the correct format.
    """
    try:
        # Convert image to OpenCV format if needed
        if isinstance(image, str):  # If it's a file path, load it
            image = cv2.imread(image)

        elif isinstance(image, Image.Image):  # If it's a PIL Image, convert it
            image = np.array(image)

        # Run YOLO classification
        results = yolo_model(image)

        for result in results:
            for box in result.boxes:
                label = result.names[int(box.cls[0])]
                # Only store CAT or DOG images
                if image_name.startswith("dog."):
                            category = "DOG"
                            class_id = 1  # Force label to DOG
                            label = "dog"
                elif image_name.startswith("cat."):
                            category = "CAT"
                            class_id = 0  # Force label to CAT
                            label = "cat" 
                else:               
                    if label == "cat":
                        category = "CAT"
                        class_id = 0  # Assign class ID for YOLO (0 for cat)
                    elif label == "dog":
                        category = "DOG"
                        class_id = 1  # Assign class ID for YOLO (1 for dog)
                    else:
                        return  # Skip non-cat/dog images

                # Extract bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Check if bbox is valid
                if x1 >= x2 or y1 >= y2:
                    print(f"⚠️ Skipping {image_name} - Invalid bounding box")
                    return

                # Crop & Resize Image
                cropped_img = image[y1:y2, x1:x2]
                cropped_img = cv2.resize(cropped_img, IMAGE_SIZE)
                cropped_img = Image.fromarray(cropped_img)

                # Save the image
                image_save_path = os.path.join(OUTPUT_DIR, category, "ORIGINAL", "images",image_name)
                cropped_img.save(image_save_path)

                # === SAVE YOLO LABEL ===
                img_width, img_height = image.shape[1], image.shape[0]

                # Convert bounding box to YOLO format (normalized)
                x_center = ((x1 + x2) / 2) / img_width
                y_center = ((y1 + y2) / 2) / img_height
                width = (x2 - x1) / img_width
                height = (y2 - y1) / img_height

                yolo_label = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

                # Save YOLO label file
                label_save_path = os.path.join(OUTPUT_DIR, category, "ORIGINAL", "labels",image_name.replace(".jpg", ".txt"))
                with open(label_save_path, "w") as f:
                    f.write(yolo_label)

                print(f"✅ Processed {image_name}: Saved image & label.")

    except Exception as e:
        print(f"❌ Error processing {image_name}: {e}")


# === FUNCTION TO CLEANUP BLUEPRINT_DS ===
def cleanup_blueprint():
    print("🗑️ Deleting existing BLUEPRINT_DS...")
    shutil.rmtree(OUTPUT_DIR)
    for category in CLASSES:
        os.makedirs(os.path.join(OUTPUT_DIR, category, "ORIGINAL"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, category, "AUGMENTED"), exist_ok=True)
        image_save_dir = os.path.join(OUTPUT_DIR, category, "ORIGINAL", "images")
        label_save_dir = os.path.join(OUTPUT_DIR, category, "ORIGINAL", "labels")
        os.makedirs(image_save_dir, exist_ok=True)
        os.makedirs(label_save_dir, exist_ok=True)        

# === MAIN EXECUTION ===
if __name__ == "__main__":
    cleanup_blueprint()  # Delete existing dataset
    # download_from_huggingface()  # Pull Hugging Face datasets
    download_from_kaggle()  # Pull Kaggle datasets
    # download_from_imagenet()  # Pull ImageNet datasets
    print("✅ BLUEPRINT_DS dataset creation complete!")
