import cv2
import os
import sys
import shutil

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))


sys.path.append(project_root)

# Now you can import using the full path from the project root
from src.services.logger_service import LoggerService


# Initialize logger with module name




# Get the absolute path of the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Add the script's directory to Python path
sys.path.append(script_dir)

config_path = os.path.join(script_dir, "config.json")
print(f"Config path: {config_path}")


from config_manager import ConfigManager
from augmentation_executor import AugmentationExecutor


def cleanup_augmentations():
    """Clear all augmented images before starting new augmentation process"""
    try:
        input_dir = config_manager.get_path("base_dir")
        for image_classification in os.listdir(input_dir):
            images_subfolders = config_manager.get_path("images_subfolders_to_use")
            for  subfolder in images_subfolders:
                output_dir =  os.path.join(input_dir,image_classification,config_manager.get_path("augmented_dir"))
        
                if os.path.exists(output_dir):
                    # Clear images directory
                    images_dir = os.path.join(output_dir, "images")
                    if os.path.exists(images_dir):
                        shutil.rmtree(images_dir)
                    os.makedirs(images_dir, exist_ok=True)
                    
                    # Clear labels directory
                    labels_dir = os.path.join(output_dir, "labels")
                    if os.path.exists(labels_dir):
                        shutil.rmtree(labels_dir)
                    os.makedirs(labels_dir, exist_ok=True)
                
        logger.info("✅ Cleared all augmentation folders")
    except Exception as e:
        logger.error(f"Failed to cleanup augmentation folders: {str(e)}")
        raise


def save_augmented_image(img, save_path):
    if img is None or img.size == 0:
        logger.error(f"Invalid image - cannot save empty image")
        return False
        
    try:
        success = cv2.imwrite(save_path, img)
        if not success:
            logger.error(f"Failed to save image to {save_path}")
            return False
        return True
    except Exception as e:
        logger.error(f"Error saving image: {str(e)}")
        return False


if __name__ == "__main__":
    config_manager = ConfigManager(config_path)
    executor = AugmentationExecutor(config_manager)

    logger = LoggerService(config_manager.get_log_file())

    cleanup_augmentations()


    input_dir = config_manager.get_path("base_dir")
    #  the folders are orgenized : 'DATASETS/BLUEPRINT_DS/<LCASIFICATION>/<DETECTION-INDICATOR>/image/<image_name>.jpg'
    for image_classification in os.listdir(input_dir):
        images_subfolders = config_manager.get_path("images_subfolders_to_use")
        for  subfolder in images_subfolders:
            output_dir =  os.path.join(input_dir,image_classification,config_manager.get_path("augmented_dir"))
            os.makedirs(output_dir, exist_ok=True)                
            
            images_path = os.path.join(input_dir, image_classification,subfolder)
            for root, dirs, files in os.walk(os.path.join(images_path,'images')):
            # Filter for image files
               for file in files:
                  image_path = os.path.join(root, file)
                  image = cv2.imread(image_path)
                  if image is None:
                        continue
                  try:
                        augmented_image, applied_augmentations = executor.apply_augmentations(image,image_path)
                        #   if len(applied_augmentations)> 0 :
                        if len(applied_augmentations)>0 and augmented_image is not None and augmented_image.size > 0:
                            try :
                                save_augmented_image(augmented_image,os.path.join(output_dir, 'images',file))
                                logger.info(f'{file} : augmentation {applied_augmentations} applied ')
                            except cv2.error as e:
                                logger.error(f'{file} : openCv error : {e} \n augmentation failed on {applied_augmentations}  ')
                        else     :
                                logger.info(f'{file} : no augmentation on file applied ')
                        #  now create a label for this file and json that it is augmentation of file...
                  except Exception as e:
                      logger.error(f'{file} : failed on apply augmentation  {e}  ')
                      
                      
                

    print("✅ Augmentation complete!")
