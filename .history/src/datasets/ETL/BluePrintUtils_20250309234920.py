import os
import shutil
import cv2
import wget
import kaggle
import numpy as np
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from ultralytics import YOLO
from torchvision import transforms
import traceback 


from services.logger_service import LoggerService

# Get the current module name without extension
module_name = os.path.splitext(os.path.basename(__file__))[0]  # This will give 'BluePrintUtils'

# Initialize logger with module name
logger = LoggerService(module_name)


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

THRESHOLD_CLASSIFICATION_CONF = 0.8 # 80% is a threshold for classsifications

DATASETS = {
    "huggingface": [
        "microsoft/cats_vs_dogs"
    ],
    "kaggle": [
        "tongpython/cat-and-dog" 
    ] 
}

OUTPUT_DIR = "./DATASETS/BLUEPRINT_DS"
CLASSES = ["CAT", "DOG"]
IMAGE_SIZE = (224, 224)

# === SETUP DIRECTORIES ===
for category in CLASSES:
    os.makedirs(os.path.join(OUTPUT_DIR, category, "ORIGINAL"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, category, "AUGMENTED"), exist_ok=True)

# === YOLO MODEL ===
cnn_model = YOLO(YOLO_MODEL)

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
def get_kaggle_label_from_file_name(file_name=""):
    if file_name.startswith("dog."):
      return "DOG"
    elif file_name.startswith("cat."):
      return "CAT"
    else :
        return ""
    
def download_from_kaggle(force_download_datasets=False, force_delete_after_load=False):
    logger.info("📥 Checking Kaggle datasets...")
    if len(DATASETS["kaggle"]) == 0:
        logger.info('no dataset found for kaggle')
        return True    
    try:
        kaggle_temp_exists = os.path.exists("kaggle_temp")
        
        # Only download if folder doesn't exist or force_download is True
        if not kaggle_temp_exists or force_download_datasets:
            if force_download_datasets and kaggle_temp_exists:
                logger.info("Force download enabled - cleaning existing kaggle_temp folder")
                shutil.rmtree("kaggle_temp")
            
            logger.info("Downloading fresh datasets from Kaggle...")
            kaggle.api.authenticate()
            for dataset_name in DATASETS["kaggle"]:
                logger.info(f"Downloading dataset: {dataset_name}")
                kaggle.api.dataset_download_files(dataset_name, path="kaggle_temp", unzip=True)
        else:
            logger.info("Using existing kaggle_temp folder (use force_download_datasets=True to force fresh download)")
        # Process downloaded images
        
        for root, _, files in os.walk("kaggle_temp"):
            for file in files:
                image_path = os.path.join(root, file)
                if file.endswith('.jpg'):
                    hard_label = get_kaggle_label_from_file_name(file)  
                    process_image(image_path, file, hard_label)
                
                
    except Exception as e:
        logger.error(f"❌ Error downloading from Kaggle: {str(e)}")
        logger.error("Please ensure your kaggle.json is properly configured")
        raise    
    finally:
        if force_delete_after_load : 
            shutil.rmtree("kaggle_temp")



# === FUNCTION TO DOWNLOAD FROM IMAGENET ===
def download_from_imagenet():
    print("📥 Downloading images from ImageNet...")
    for category, synset in DATASETS["imagenet"].items():
        url = f"http://www.image-net.org/api/text/imagenet.synset.geturls?wnid={synset}"
        urls = wget.download(url).split("\n")

        for idx, img_url in tqdm(enumerate(urls), total=len(urls)):
            try:
                image_path = f"imagenet_{category}_{idx}.jpg"
                wget.download( img_url, image_path)
                process_image(image_path, image_path)
            except:
                continue  # Skip broken images

# === FUNCTION TO PROCESS IMAGE (YOLO CLASSIFICATION) ===
def process_image(image, image_name, hard_label = ""):
    """
    Runs YOLO on the image, classifies it as CAT or DOG, and saves it to BLUEPRINT_DS.
    Also, saves YOLO labels in the correct format.
    """
    if not hard_label  : 
        raise "type of image in the dataset mentioned as hard_label - is mandatory"
    try:
        # Convert image to OpenCV format if needed
        if isinstance(image, str):  # If it's a file path, load it
            image = cv2.imread(image)

        elif isinstance(image, Image.Image):  # If it's a PIL Image, convert it
            image = np.array(image)

        # Run YOLO classification
        results = cnn_model(image)

        for result in results:
            for box in result.boxes:
                label = result.names[int(box.cls[0])]
                origin_indicator = "ORIGINAL"
                if label == "cat":
                        category = "CAT"
                        class_id = 0  # Assign class ID for YOLO (0 for cat)
                elif label == "dog":
                        category = "DOG"
                        class_id = 1  # Assign class ID for YOLO (1 for dog)
                else:
                    logger.info (f"the label is not recognized ({image_name})")
                    return  # Skip non-cat/dog images
                

                #  WE FILTE ONLY CONFIDENCE PICTURESD GET TO LOAD TO DATASET
                confidence = float(result.boxes.conf.cpu().numpy().max())
                if category != hard_label:
                    origin_indicator = "FALSE_POSITIVE"
                elif confidence < THRESHOLD_CLASSIFICATION_CONF : 
                    origin_indicator = "LOW_PREDICTION"

                # Extract bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Check if bbox is valid
                if x1 >= x2 or y1 >= y2:
                    logger.warning(f"⚠️ Skipping {image_name} - Invalid bounding box")
                    return

                # Crop & Resize Image
                cropped_img = image[y1:y2, x1:x2]
                cropped_img = cv2.resize(cropped_img, IMAGE_SIZE)
                cropped_img = Image.fromarray(cropped_img)

                # Save the image
                
                
                image_save_path = os.path.join(OUTPUT_DIR, category, origin_indicator, "images",image_name)
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

                logger.info(f"✅ Processed {image_name}: Saved image & label.")

    except Exception as e:
        logger.error(f"❌ Error processing {image_name}: {e}")
        print("Full traceback:")
        print(traceback.format_exc())  # This will print the full stack trace with line numbers
        print("Please ensure your kaggle.json is properly configured")
            


# === FUNCTION TO CLEANUP BLUEPRINT_DS ===
def cleanup_blueprint():
    print("🗑️ Deleting existing BLUEPRINT_DS...")
    shutil.rmtree(OUTPUT_DIR)
    for category in CLASSES:
        os.makedirs(os.path.join(OUTPUT_DIR, category, "ORIGINAL"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, category, "AUGMENTED"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, category, "FALSE_POSITIVE"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, category, "LOW_PREDICTION"), exist_ok=True)
        image_save_dir = os.path.join(OUTPUT_DIR, category, "ORIGINAL", "images")
        label_save_dir = os.path.join(OUTPUT_DIR, category, "ORIGINAL", "labels")
        os.makedirs(image_save_dir, exist_ok=True)
        os.makedirs(label_save_dir, exist_ok=True)        
        image_save_dir = os.path.join(OUTPUT_DIR, category, "LOW_PREDICTION", "images")
        label_save_dir = os.path.join(OUTPUT_DIR, category, "LOW_PREDICTION", "labels")
        os.makedirs(image_save_dir, exist_ok=True)
        os.makedirs(label_save_dir, exist_ok=True)        
        image_save_dir = os.path.join(OUTPUT_DIR, category, "FALSE_POSITIVE", "images")
        label_save_dir = os.path.join(OUTPUT_DIR, category, "FALSE_POSITIVE", "labels")
        os.makedirs(image_save_dir, exist_ok=True)
        os.makedirs(label_save_dir, exist_ok=True)     
        image_save_dir = os.path.join(OUTPUT_DIR, category, "AUGMENTED", "images")
        label_save_dir = os.path.join(OUTPUT_DIR, category, "AUGMENTED", "labels")
        os.makedirs(image_save_dir, exist_ok=True)
        os.makedirs(label_save_dir, exist_ok=True)     
        
           

# === MAIN EXECUTION ===
if __name__ == "__main__":
    cleanup_blueprint()  # Delete existing dataset
    # download_from_huggingface()  # Pull Hugging Face datasets
    download_from_kaggle(force_download_datasets=False, force_delete_after_load=False)  # Pull Kaggle datasets
    # download_from_imagenet()  # Pull ImageNet datasets
    print("✅ BLUEPRINT_DS dataset creation complete!")
    logger.info("✅ BLUEPRINT_DS dataset creation complete!")
