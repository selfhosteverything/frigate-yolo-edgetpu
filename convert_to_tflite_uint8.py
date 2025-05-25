import tensorflow as tf
import os
import cv2
import numpy as np
import argparse
import glob

# Example letterbox function (you might need to adjust based on Ultralytics' exact one)
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleUp=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleUp:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

def representative_dataset_gen():
    folder_path = ARGS.dataset_path
    print(f"Generating representative dataset from: {folder_path}")
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    num_calibration_images = min(ARGS.num_calib_images, len(image_files))
    
    for i in range(num_calibration_images):
        image_path = image_files[i]
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read image {image_path}. Skipping.")
            continue
        
        # Preprocess: Letterbox, then normalize to [0,1] float32
        img_resized, _, _ = letterbox(image, new_shape=(ARGS.img_height, ARGS.img_width), auto=False) # auto=False for exact square, adjust if needed
        normalized_image = img_resized.astype(np.float32) / 255.0
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
