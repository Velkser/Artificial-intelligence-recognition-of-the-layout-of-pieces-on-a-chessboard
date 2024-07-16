# %%
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf

from PIL import Image
from shutil import copyfile
from keras.callbacks import EarlyStopping

from tensorflow.keras.preprocessing import image # type: ignore

from tensorflow.keras import regularizers, layers, preprocessing # type: ignore
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D, GlobalMaxPooling2D, BatchNormalization, Lambda # type: ignore
from tensorflow.keras.models import Sequential, load_model, Model # type: ignore

from tensorflow.keras.optimizers import Adam # type: ignore

from tensorflow.keras.applications import VGG16, ResNet101 # type: ignore
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input # type: ignore
from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocess_input # type: ignore

from tensorflow.keras.utils import image_dataset_from_directory # type: ignore
from tensorflow.keras.losses import SparseCategoricalCrossentropy # type: ignore

from preprocessing_data import load_data, change_data, change_data_hierarchy, create_first_dataset, create_second_dataset
from corner_and_sides_detection import create_variables_for_png_files

import pandas as pd

def show_images(images):
    fig, axes = plt.subplots(1, len(images), figsize=(30, 30))
    for ax, image in zip(axes, images):
        image = image / 255.0
        ax.imshow(image)
        ax.axis('off')
    plt.show()
    
def replace_values(tensor):
    return tf.where(tensor != 2, 1, 0)
    

def create_augmentation_layer():
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(0.2),
    ])


def create_vgg16_pretrained_model(data_augmentation, vgg16_preprocess_input, ds_train_first, ds_test_first, ds_val_first):
    base_model_vgg16 = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )

    base_model_vgg16.trainable = False
    
    inputs_vgg16 = Input(shape=(224, 224, 3))
    x_vgg16 = data_augmentation(inputs_vgg16)
    x_vgg16 = Lambda(vgg16_preprocess_input)(inputs_vgg16)
    x_vgg16 = base_model_vgg16(x_vgg16, training=False)
    x_vgg16 = Conv2D(filters=256, kernel_size=3, activation='relu', padding='same')(x_vgg16)
    x_vgg16 = GlobalAveragePooling2D()(x_vgg16)
    x_vgg16 = Dropout(0.2)(x_vgg16)
    x_vgg16 = BatchNormalization()(x_vgg16)
    outputs_vgg16 = Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01))(x_vgg16)
    first_model = Model(inputs_vgg16, outputs_vgg16)


    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=3,
        min_delta = 0.001,           # minimium amount of change to count as an improvement
        restore_best_weights = True, # restore the weights to model with the lowest validation loss
    )

    base_learning_rate = 0.0001
    optimizer = tf.keras.optimizers.Adam(epsilon=base_learning_rate)

    first_model.compile(optimizer=optimizer,
                        loss=tf.keras.losses.BinaryCrossentropy(),
                        metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.5, name='accuracy')])

    initial_epochs = 20
    history = first_model.fit(ds_train_first, epochs=initial_epochs,
                        validation_data = ds_val_first, batch_size=32,
                        callbacks=[early_stopping],)

    loss0, accuracy0 = first_model.evaluate(ds_val_first)


    base_model_vgg16.trainable = True

    for layer in base_model_vgg16.layers[:-5]:
        layer.trainable = False

    first_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate/10),
                    loss=tf.keras.losses.BinaryCrossentropy(),
                   metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.5, name='accuracy')])

    fine_tune_epochs = 10
    total_epochs =  initial_epochs + fine_tune_epochs
    history_fine = first_model.fit(ds_train_first, epochs=total_epochs,initial_epoch=history.epoch[-1],
                        validation_data = ds_val_first, batch_size=32,
                        callbacks=[early_stopping],)

    first_model.save('first_model.keras')

    loss, accuracy = first_model.evaluate(ds_test_first)
    print('Test accuracy for vgg16 pretrained model :', accuracy)

def create_resnet101_pretrained_model(data_augmentation, resnet_preprocess_input, ds_train_second, ds_test_second, ds_val_second):
    base_model_resnet101 = ResNet101(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3))

    base_model_resnet101.trainable = False

    inputs_resnet101 = Input(shape=(224, 224, 3))
    x_resnet101 = Lambda(resnet_preprocess_input)(inputs_resnet101)
    x_resnet101 = base_model_resnet101(x_resnet101, training=False)
    x_resnet101 = BatchNormalization()(x_resnet101)
    x_resnet101 = Conv2D(filters=256, kernel_size=3, activation='relu', padding='same')(x_resnet101)
    x_resnet101 = GlobalMaxPooling2D()(x_resnet101)
    x_resnet101 = BatchNormalization()(x_resnet101)
    x_resnet101 = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x_resnet101)
    x_resnet101 = Dropout(0.3)(x_resnet101)
    outputs_resnet101 = Dense(12, activation='softmax', kernel_regularizer=regularizers.l2(0.01))(x_resnet101)
    second_model = Model(inputs_resnet101, outputs_resnet101)


    base_learning_rate = 0.0001
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                patience=3, min_delta=1e-3,
                                                restore_best_weights = True)

    optimizer = tf.keras.optimizers.Adam(epsilon=base_learning_rate)

    second_model.compile(optimizer=optimizer,
                        loss=SparseCategoricalCrossentropy(from_logits=False),
                        metrics=['sparse_categorical_accuracy'])

    initial_epochs = 20
    loss, accuracy0 = second_model.evaluate(ds_val_second)

    history = second_model.fit(ds_train_second, 
                            epochs=initial_epochs,
                            batch_size=32,
                            validation_data=ds_val_second, 
                            callbacks=[callback])

    base_model_resnet101.trainable = True

    for layer in base_model_resnet101.layers[:-30]:
        layer.trainable = False

    second_model.compile(optimizer=tf.keras.optimizers.Adam(epsilon=base_learning_rate/10),
                        loss=SparseCategoricalCrossentropy(from_logits=False),
                        metrics=['sparse_categorical_accuracy'])

    fine_tune_epochs = 10
    total_epochs =  initial_epochs + fine_tune_epochs

    history_fine = second_model.fit(ds_train_second, 
                            epochs=initial_epochs,
                            initial_epoch=history.epoch[-1],
                            batch_size=32,
                            validation_data=ds_val_second, 
                            callbacks=[callback])

    second_model.save('second_model.keras')
    
def load_keras_model(filepath, type = 'vgg16'):
    """
    Loads a Keras model from a specified .h5 file.

    Args:
    filepath (str): Path to the .keras file containing the saved Keras model.

    Returns:
    keras.Model: Loaded Keras model.

    Raises:
    FileNotFoundError: If the specified file does not exist.
    OSError: If the file is not a valid keras file or is corrupted.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"The file {filepath} does not exist.")

    try:
        if type == 'vgg16':
            preprocess_input_custom = preprocess_input_custom_vgg16
        else:
            preprocess_input_custom = preprocess_input_custom_resnet

        model = load_model(filepath, custom_objects={'preprocess_input': preprocess_input_custom})
        print("Model loaded successfully.")
        return model
    except OSError as e:
        print("Failed to load the model. The file may be corrupted or not a valid Keras file.")
        raise e
    
@tf.keras.utils.register_keras_serializable()
def preprocess_input_custom_vgg16(x):
    return vgg16_preprocess_input(x)

@tf.keras.utils.register_keras_serializable()
def preprocess_input_custom_resnet(x):
    return resnet_preprocess_input(x)

def predict_image_class(model, image_path, binary = True):
    """
    Accepts a Keras model and a path to an image, loads and preprocesses the image,
    makes a prediction using the model, and returns the index of the class with the highest probability.

    Args:
    model (tf.keras.Model): Trained Keras model to be used for prediction.
    image_path (str): Path to the image to be classified.
    binary (bool): Binary classification or not
    
    Returns:
    int: Prediction result.
    """
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)  # Creating a batch

    # Prediction
    prediction = model.predict(img_batch)
    if binary:
    # Round the prediction to nearest integer (0 or 1)
        return np.round(prediction).astype(int)[0][0]
    else:
        return np.argmax(prediction)


def find_png_json_pairs(folder_path):
    """
    Returns a list of tuples, each containing paths to a .jpg file and its corresponding .json file within the specified folder.

    Args:
    folder_path (str): Path to the folder where .jpg files and their corresponding .json files are searched.

    Returns:
    list of tuples: A list of tuples, where each tuple contains paths to a .jpg file and its corresponding .json file.
    """
    file_pairs = []
    if not os.path.exists(folder_path):
        print("The specified folder does not exist.")
        return file_pairs

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.jpg'):
                png_path = os.path.join(root, file)
                json_path = os.path.splitext(png_path)[0] + '.json'
                if os.path.exists(json_path):
                    file_pairs.append((png_path, json_path))
                else:
                    print(f"Warning: No corresponding JSON file found for {png_path}")

    return file_pairs


def compare_predictions_with_ground_truth(model, model2, file_pairs, class_names, binary=True ):
    results = []

    for img_path, json_path in file_pairs:
        if binary:
            predicted_class = predict_image_class(model, img_path, binary=True)
            if predicted_class == 0:
                res = "empty"
        else:
            predicted_class = predict_image_class(model, img_path, binary=True)
            if predicted_class == 1:
                predicted_class = predict_image_class(model2, img_path, binary=False)
                res = class_names[predicted_class]
            else:
                res = "empty"

        with open(json_path, 'r') as file:
            actual_class = json.load(file)['value']
        
        results.append({'Actual': actual_class, 'Predicted': res})

    results_df = pd.DataFrame(results)

    return results_df

def is_valid_chess_move(start, end, model1, model2, image_folder):
    """
    Check if a chess move from `start` to `end` is valid based on the rules of chess pieces movement and capturing,
    using two models for detection.
    
    Args:
    - start (str): Chessboard starting position (e.g., 'a2').
    - end (str): Chessboard ending position (e.g., 'a3').
    - model1 (tf.keras.Model): Model to detect presence of a piece.
    - model2 (tf.keras.Model): Model to classify the type of piece.
    - image_folder (str): Directory path containing images of positions.
    
    Returns:
    - dict: Information about the move validity and the outcome.
    """
    piece_classes = ['b_black', 'b_white', 'k_black', 'k_white', 'n_black', 'n_white', 'p_black', 'p_white', 'q_black', 'q_white', 'r_black', 'r_white']
    
    def piece_movement(piece, start, end, color):
        """Define movement rules for each piece type."""
        start_col, start_row = ord(start[0]) - ord('a'), int(start[1])
        end_col, end_row = ord(end[0]) - ord('a'), int(end[1])
        col_diff = end_col - start_col
        row_diff = end_row - start_row

        if piece == 'p':  # Pawn
            if color == 'white':
                if start_row == 2 and end_row == 4 and start_col == end_col and not is_piece_present(start_col, 3):  # Initial two-step move
                    return True
                return end_row - start_row == 1 and (abs(col_diff) == 1 or start_col == end_col)
            else:  # Black pawn
                if start_row == 7 and end_row == 5 and start_col == end_col and not is_piece_present(start_col, 6):  # Initial two-step move
                    return True
                return start_row - end_row == 1 and (abs(col_diff) == 1 or start_col == end_col)

        elif piece == 'r':  # Rook
            return start_col == end_col or start_row == end_row  # Straight line movement

        elif piece == 'n':  # Knight
            return (abs(col_diff) == 2 and abs(row_diff) == 1) or (abs(col_diff) == 1 and abs(row_diff) == 2)  # L-shape

        elif piece == 'b':  # Bishop
            return abs(col_diff) == abs(row_diff)  # Diagonal movement

        elif piece == 'q':  # Queen
            return abs(col_diff) == abs(row_diff) or start_col == end_col or start_row == end_row  # Combine rook and bishop

        elif piece == 'k':  # King
            return max(abs(col_diff), abs(row_diff)) == 1  # One square in any direction

        return False

    def is_piece_present(col, row):
        """Check if there is a piece on the given position using model1."""
        img_path = os.path.join(image_folder, f"{chr(ord('a') + col)}{row}.jpg")
        return predict_image_class(model1, img_path, binary=True)
    
    def check_path_clear(start, end, model1, image_folder):
        """
        Check if the path between start and end positions on a chessboard is clear of any pieces.
        
        Args:
        - start (str): Start position (e.g., 'a1').
        - end (str): End position (e.g., 'h8').
        - model1 (tf.keras.Model): Model to detect presence of a piece.
        - image_folder (str): Path to the folder containing chess square images.
        
        Returns:
        - bool: True if the path is clear, False otherwise.
        """
        start_col, start_row = ord(start[0]) - ord('a'), int(start[1])
        end_col, end_row = ord(end[0]) - ord('a'), int(end[1])
        
        # Determine the direction of movement
        col_step = 1 if start_col < end_col else -1 if start_col > end_col else 0
        row_step = 1 if start_row < end_row else -1 if start_row > end_row else 0
        
        current_col = start_col + col_step
        current_row = start_row + row_step
        
        # Move step by step from start to end, not including end
        while current_col != end_col or current_row != end_row:
            img_path = os.path.join(image_folder, f"{chr(ord('a') + current_col)}{current_row}.jpg")
            if predict_image_class(model1, img_path, binary=True):
                return False  # There is a piece in the path
            current_col += col_step
            current_row += row_step
        
        return True

    start_img_path = os.path.join(image_folder, f"{start}.jpg")
    end_img_path = os.path.join(image_folder, f"{end}.jpg")
    
    # Check if there is a piece at the start position
    start_piece_present = predict_image_class(model1, start_img_path, binary=True)
    if not start_piece_present:
        return {"valid": False, "reason": "No piece at start position"}
    
    # Identify the piece at start position
    start_piece_type = predict_image_class(model2, start_img_path, binary=False)
    piece_type = piece_classes[start_piece_type]
    color = 'white' if 'white' in piece_type else 'black'
    piece = piece_type[0].lower()
    
    # Check if the move is possible for the piece type
    if not piece_movement(piece, start, end, color):
        return {"valid": False, "reason": "Invalid move for piece type"}
    
    # Check if the path is clear for non-knight pieces
    if piece in ['q', 'r', 'b'] and not check_path_clear(start, end,  model1, image_folder):
        return {"valid": False, "reason": "Path is not clear"}

    # Check if capturing or just moving
    end_piece_present = predict_image_class(model1, end_img_path, binary=True)
    if end_piece_present:
        end_piece_type = predict_image_class(model2, end_img_path, binary=False)
        capture_piece = piece_classes[end_piece_type]
        if 'white' in piece_type and 'white' in capture_piece or 'black' in piece_type and 'black' in capture_piece:
            return {"valid": False, "reason": "Cannot capture own piece"}
        else:
            return {"valid": True, "move_type": "capture", "captured": capture_piece}
    
    return {"valid": True, "move_type": "move", "captured": None}





#!The part that is responsible for loading datasets, this part does not work if there are no folders with files themselves
# script_directory = os.getcwd()

# batch_size = 32
# img_height = 224
# img_width = 224
# data_dir = os.path.join(script_directory, "your data")

# ds_train_first, ds_test_first, ds_val_first = create_first_dataset(data_dir)
# ds_train_second, ds_test_second, ds_val_second = create_second_dataset(data_dir)

# AUTOTUNE = tf.data.AUTOTUNE

# ds_train_first = ds_train_first.cache().prefetch(buffer_size=AUTOTUNE)
# ds_val_first = ds_val_first.cache().prefetch(buffer_size=AUTOTUNE)
# ds_test_first = ds_test_first.cache().prefetch(buffer_size=AUTOTUNE)


# ds_train_second = ds_train_second.cache().prefetch(buffer_size=AUTOTUNE)
# ds_val_second = ds_val_second.cache().prefetch(buffer_size=AUTOTUNE)
# ds_test_second = ds_test_second.cache().prefetch(buffer_size=AUTOTUNE)


# data_augmentation = create_augmentation_layer()

# ds_val_first = ds_val_first.map(lambda image_batch, labels_batch: (image_batch, replace_values(labels_batch))).batch(batch_size)
# ds_train_first = ds_train_first.map(lambda image_batch, labels_batch: (image_batch, replace_values(labels_batch))).batch(batch_size)
# ds_test_first = ds_test_first.map(lambda image_batch, labels_batch: (image_batch, replace_values(labels_batch))).batch(batch_size)


# ds_train_second = ds_train_second.batch(batch_size)
# ds_val_second = ds_val_second.batch(batch_size)
# ds_test_second = ds_test_second.batch(batch_size)


# #!The part that is responsible for testing already trained models

# model1 = load_keras_model("first_model.keras")
# model2 = load_keras_model("second_model.keras", type = 'resnet')

# test1 = predict_image_class(model1, 'app_data_split/1406/b2.jpg', True)
# test2 = predict_image_class(model2, 'app_data_split/1406/b2.jpg', False)

# class_names = ['b_black', 'b_white', 'k_black', 'k_white', 'n_black', 'n_white', 'p_black', 'p_white', 'q_black', 'q_white', 'r_black', 'r_white']
# print(f"Res for test1 = {test1}, res for test2  {test2}")

# import time
# start_time = time.time()
# file_pairs = find_png_json_pairs("app_data_split/1406")
# test3 = compare_predictions_with_ground_truth(model1, model2, file_pairs, class_names, False)
# elapsed_time = time.time() - start_time
# print(f"Execution time: {elapsed_time} seconds")
# print(f"Res for test3 = {test3}")


# test4 = is_valid_chess_move("b2", "b7", model1, model2, "app_data_split/1406")
# print(f"Res for test4 = {test4}")