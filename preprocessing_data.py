import os
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf

from shutil import copyfile

import shutil
import tempfile

def load_data(folder_path, height, width, channels):
    """
    Load image data and corresponding JSON metadata from a directory structure.

    Args:
        folder_path (str): Path to the folder containing image and JSON files.
        height (int): The height to which images are resized.
        width (int): The width to which images are resized.
        channels (int): The number of image channels.

    Returns:
        tuple: Two numpy arrays containing the images (x_train) and their corresponding labels (y_train).
    """
    # Initialize lists for x_train and y_train
    x_train = []
    y_train = []

    # Recursively traverse all subdirectories in the specified directory
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            if file_name.endswith(".jpg"):
                # Get the path to the image
                image_path = os.path.join(root, file_name)
                # Load the image and resize it
                image = cv2.imread(image_path)
                image = cv2.resize(image, (width, height))
                # Add the image to the x_train list
                x_train.append(image)

                # Get the path to the corresponding JSON file
                json_path = os.path.join(root, file_name[:-4] + ".json")
                # Load the JSON file and extract the value of the key "value"
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    value = data.get("value", None)
                # Add the key value to the y_train list
                y_train.append(value)

    # Convert lists to numpy arrays
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    return x_train, y_train

def change_data(folder_path, height, width, channels, save_folder = "new_data"):
    """
    Processes images and their corresponding JSON files to resize and categorize them by content into new directories.

    Args:
        folder_path (str): Path to the folder containing image and JSON files.
        height (int): The height to which images are resized.
        width (int): The width to which images are resized.
        channels (int): The number of image channels.

    This function reads each image and its corresponding JSON, resizes the image,
    and then stores it in a new directory structure based on the content specified in the JSON file.
    """
    # Get the path to the current script directory
    script_dir = os.getcwd()
    # Path to the folder where the images will be saved
    # save_folder = os.path.join(script_dir, 'new_data')
    # print(save_folder)
    # Recursively traverse all subdirectories in the specified directory
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            if file_name.endswith(".jpg"):
                # Get the path to the image
                image_path = os.path.join(root, file_name)
                # Load the image and resize it
                image = cv2.imread(image_path)
                image = cv2.resize(image, (width, height))

                # Get the path to the corresponding JSON file
                json_path = os.path.join(root, file_name[:-4] + ".json")
                # Load the JSON file and extract the value of the key "value"
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    value = data.get("value", None)

                # Determine the save path depending on the class of the piece
                if value.lower() == 'empty':
                    save_path = os.path.join(save_folder, "empty")
                else:
                    # Determine the color of the piece
                    save_color = 'white' if value.isupper() else 'black'
                    # Create a folder for each chess piece and its color if it doesn't already exist
                    save_path = os.path.join(save_folder, value.lower() + '_' + save_color)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                # Copy the image to the folder with a modified name
                new_image_name = f"{file_name[:-4]}_{len(os.listdir(save_path))}.jpg"
                new_image_path = os.path.join(save_path, new_image_name)

                copyfile(image_path, new_image_path)
                
def change_data_hierarchy(script_directory=os.getcwd(), data_exists=True):
    """
    Checks if data has already been processed and organized. If not, processes and organizes training and validation data.

    Args:
    script_directory (str): The directory of the script or specified directory.
    data_exists (bool): Flag indicating whether the data has already been processed.

    This function reorganizes image data into structured directories for easier access and processing for training.
    """
    if not data_exists:
        try:
            # Paths to the train and validation directories
            folder_path_train = os.path.join(script_directory, "data", "train")
            folder_path_val = os.path.join(script_directory, "data", "val")

            # Default image dimensions and channels
            height, width, channels = 224, 224, 3

            # Process and organize train and validation data
            change_data(folder_path_train, height, width, channels)
            change_data(folder_path_val, height, width, channels)

            # Update the flag to indicate data has been processed
            data_exists = True
        except KeyboardInterrupt:
            # Handle the interruption and keep the flag unchanged
            print("Script was manually interrupted, the 'data_exists' variable was not changed")

            
def create_first_dataset(data_dir, img_width=224, img_height=224, batch_size=32):
    """
    Creates datasets for training, testing, and validation from the specified directory.

    Args:
        data_dir (str): Directory containing image data.
        img_width (int): Width of the images after resizing.
        img_height (int): Height of the images after resizing.
        batch_size (int): Batch size for dataset processing.

    Returns:
        Tuple of datasets (ds_train_first, ds_test_first, ds_val_first) for training, testing, and validation respectively.
    """
    # Load the training data without batching
    ds_train_first = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=None,
    )

    # Load the validation data without batching
    ds_val_first = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=None,
    )

    # Determine the number of validation batches
    val_batches = tf.data.experimental.cardinality(ds_val_first)
    # Take a portion of the validation set to create a test set
    ds_test_first = ds_val_first.take(val_batches // 5)
    # Skip the same portion to adjust the validation set
    ds_val_first = ds_val_first.skip(val_batches // 5)

    return ds_train_first, ds_test_first, ds_val_first

def create_second_dataset(data_dir, folder_to_exclude="new_data/empty", img_width=224, img_height=224, batch_size=32):
    """
    Creates training, testing, and validation datasets from the specified directory, 
    temporarily excluding a specified folder during dataset creation.

    Args:
        data_dir (str): Directory containing the image data.
        folder_to_exclude (str): Subdirectory within data_dir to temporarily exclude.
        img_width (int): Width of the images after resizing.
        img_height (int): Height of the images after resizing.
        batch_size (int): Batch size for dataset processing (not used in dataset creation here).

    Returns:
        Tuple of datasets (ds_train_second, ds_test_second, ds_val_second) for training, testing, and validation respectively.
    """
    try:
        # Create a temporary directory and move the folder to exclude
        temp_dir = tempfile.mkdtemp()
        folder_to_move = os.path.join(data_dir, folder_to_exclude)
        shutil.move(folder_to_move, temp_dir)

        # Load the validation data without batching
        ds_val_second = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=None
        )

        # Load the training data without batching
        ds_train_second = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=None
        )

        # Determine the number of validation batches
        val_batches = tf.data.experimental.cardinality(ds_val_second)
        # Take a portion of the validation set to create a test set
        ds_test_second = ds_val_second.take(val_batches // 5)
        # Skip the same portion to adjust the validation set
        ds_val_second = ds_val_second.skip(val_batches // 5)

    finally:
        # Move the excluded folder back to its original location and clean up the temporary directory
        shutil.move(os.path.join(temp_dir, os.path.basename(folder_to_exclude)), folder_to_move)
        shutil.rmtree(temp_dir)

    return ds_train_second, ds_test_second, ds_val_second

