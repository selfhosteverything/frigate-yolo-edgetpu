import tensorflow as tf
import os
import cv2
import numpy as np
import argparse
import glob

def representative_dataset_gen():
    folder_path = ARGS.dataset_path # This comes from command line arguments
    print(f"Generating representative dataset from: {folder_path}")
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    # Limit number of images for calibration (e.g., 100-200)
    num_calibration_images = min(ARGS.num_calib_images, len(image_files))
    
    for i in range(num_calibration_images):
        image_path = image_files[i]
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read image {image_path}. Skipping.")
            continue
        
        # Preprocess: Resize and normalize to [0,1] float32
        # This is because the SavedModel is likely float32 and expects normalized input
        resized_image = cv2.resize(image, (ARGS.img_width, ARGS.img_height))
        normalized_image = resized_image.astype(np.float32) / 255.0
        input_tensor = np.expand_dims(normalized_image, axis=0)
        yield [input_tensor]
    print(f"Finished generating representative dataset with {num_calibration_images} images.")

def main():
    saved_model_dir = ARGS.saved_model_path
    output_tflite_file = ARGS.output_tflite_path

    if not os.path.isdir(saved_model_dir):
        print(f"Error: SavedModel directory not found at {saved_model_dir}")
        exit(1)
    if not os.path.isdir(ARGS.dataset_path):
        print(f"Error: Dataset directory not found at {ARGS.dataset_path}")
        exit(1)

    print(f"Converting SavedModel from: {saved_model_dir}")
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

    # --- Full Integer Quantization (INT8) with UINT8 Input/Output ---
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    
    # CRITICAL: Set the target input type for the TFLite model to UINT8
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    # Ensure ops are compatible with INT8 TFLite builtins (good for EdgeTPU)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]
    # If the above fails due to ops not having an INT8 kernel, you might need:
    # converter.target_spec.supported_ops = [
    #     tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
    #     tf.lite.OpsSet.TFLITE_BUILTINS # To allow float ops if some can't be int8
    # ]
    # But this might mean not all ops run on EdgeTPU.

    print("Starting TFLite conversion with quantization...")
    try:
        tflite_quant_model = converter.convert()
        print("Conversion successful!")
    except Exception as e:
        print(f"Error during TFLite conversion: {e}")
        exit(1)

    with open(output_tflite_file, 'wb') as f:
        f.write(tflite_quant_model)
    print(f"Quantized TFLite model (expecting UINT8 input) saved to: {output_tflite_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Quantize a TensorFlow SavedModel to TFLite (INT8 input/output).')
    parser.add_argument('saved_model_path', type=str, help='Path to the TensorFlow SavedModel directory.')
    parser.add_argument('dataset_path', type=str, help='Path to the image directory for representative dataset.')
    parser.add_argument('output_tflite_path', type=str, help='Path to save the quantized .tflite model.')
    parser.add_argument('--img_height', type=int, required=True, help='Target image height for representative dataset and model input.')
    parser.add_argument('--img_width', type=int, required=True, help='Target image width for representative dataset and model input.')
    parser.add_argument('--num_calib_images', type=int, default=100, help='Number of images for representative dataset.')
    
    ARGS = parser.parse_args()
    main()
