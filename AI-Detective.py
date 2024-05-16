#!/usr/bin/env python3

import os
import argparse
import shutil
import subprocess
from PIL import Image
import numpy as np

# ASCII art
ascii_art = r"""
  ______   ______        _______              __                            __      __                                         ______         __   
 /      \ |      \      |       \            |  \                          |  \    |  \                                       /      \      _/  \  
|  $$$$$$\ \$$$$$$      | $$$$$$$\  ______  _| $$_     ______    _______  _| $$_    \$$ __     __   ______         __     __ |  $$$$$$\    |   $$  
| $$__| $$  | $$ ______ | $$  | $$ /      \|   $$ \   /      \  /       \|   $$ \  |  \|  \   /  \ /      \       |  \   /  \| $$$\| $$     \$$$$  
| $$    $$  | $$|      \| $$  | $$|  $$$$$$\\$$$$$$  |  $$$$$$\|  $$$$$$$ \$$$$$$  | $$ \$$\ /  $$|  $$$$$$\       \$$\ /  $$| $$$$\ $$      | $$  
| $$$$$$$$  | $$ \$$$$$$| $$  | $$| $$    $$ | $$ __ | $$    $$| $$        | $$ __ | $$  \$$\  $$ | $$    $$        \$$\  $$ | $$\$$\$$      | $$  
| $$  | $$ _| $$_       | $$__/ $$| $$$$$$$$ | $$|  \| $$$$$$$$| $$_____   | $$|  \| $$   \$$ $$  | $$$$$$$$         \$$ $$  | $$_\$$$$ __  _| $$_ 
| $$  | $$|   $$ \      | $$    $$ \$$     \  \$$  $$ \$$     \ \$$     \   \$$  $$| $$    \$$$    \$$     \          \$$$    \$$  \$$$|  \|   $$ \
 \$$   \$$ \$$$$$$       \$$$$$$$   \$$$$$$$   \$$$$   \$$$$$$$  \$$$$$$$    \$$$$  \$$     \$      \$$$$$$$           \$      \$$$$$$  \$$ \$$$$$$
                                                                                                                                                   
                                                                                                                                                   
                                                                                                                                                   


            AI Detective v0.1 - SalvatoreDeF
"""



# Function to check if system is Windows
def is_windows():
    return os.name == "nt"

# Function to install dependencies using pip
def install_dependencies():
    required_packages = ['Pillow', 'tensorflow', 'halo']
    missing_packages = [pkg for pkg in required_packages if not is_module_installed(pkg)]
    if missing_packages:
        print(f"Installing missing dependencies: {', '.join(missing_packages)}")
        for pkg in missing_packages:
            install_module(pkg)

# Function to check if a module is installed
def is_module_installed(module_name):
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False

# Function to install a module using pip
def install_module(module_name):
    try:
        subprocess.run(["pip", "install", module_name], check=True)
    except subprocess.CalledProcessError:
        print(f"Failed to install {module_name}. Please install it manually.")

# Check and install halo if missing
if not is_module_installed('halo'):
    install_module('halo')

# Import halo after installation
from halo import Halo

# Function to create directories
def create_directories(input_dir, output_file):
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Function to copy pre-trained model
def copy_pretrained_model(model_path):
    model_dir = os.path.join(os.getcwd(), "model")
    os.makedirs(model_dir, exist_ok=True)
    shutil.copy(model_path, os.path.join(model_dir, "ai_image_detection_model.h5"))

# Load pre-trained model
def load_model():
    model_path = os.path.join(os.getcwd(), "model", "ai_image_detection_model.h5")
    if os.path.exists(model_path):
        import tensorflow as tf
        return tf.keras.models.load_model(model_path)
    else:
        print("Error: Pre-trained model not found.")
        exit(1)

# Function to preprocess image
def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.resize((256, 256))  # Resize image to match model input size
    image_array = np.array(image) / 255.0  # Normalize pixel values
    return np.expand_dims(image_array, axis=0)  # Add batch dimension

# Function to detect AI image
def detect_ai_image(model, image_path):
    preprocessed_image = preprocess_image(image_path)
    prediction = model.predict(preprocessed_image)
    confidence_score = prediction[0][0]  # Assuming binary classification (0: Real, 1: AI-generated)
    confidence_percentage = round(confidence_score * 100, 2)
    return confidence_percentage

# Function to process directory
def process_directory(model, directory, output_file):
    with open(output_file, 'w') as f:
        f.write("Image Path,Confidence (%)\n")  # Write header
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(('.jpg', '.png', '.bmp')):
                    image_path = os.path.join(root, file)
                    confidence_percentage = detect_ai_image(model, image_path)
                    result = f"{image_path},{confidence_percentage}\n"
                    f.write(result)

def main():
    parser = argparse.ArgumentParser(description='Detect AI-generated images')
    parser.add_argument('input_directory', type=str, help='Path to the input directory containing images')
    parser.add_argument('output_file', type=str, help='Path to the output file for results')
    parser.add_argument('--model_path', type=str, default=None, help='Path to the pre-trained model file')
    args = parser.parse_args()

    install_dependencies()
    create_directories(args.input_directory, args.output_file)

    if args.model_path:
        copy_pretrained_model(args.model_path)
        model = load_model()
    else:
        model = load_model()

    process_directory(model, args.input_directory, args.output_file)

if __name__ == '__main__':
    main()
