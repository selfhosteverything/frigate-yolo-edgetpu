import logging
import os

import numpy as np
import cv2
from pydantic import Field
from typing_extensions import Literal

from frigate.detectors.detection_api import DetectionApi
from frigate.detectors.detector_config import BaseDetectorConfig, ModelTypeEnum

try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
except ModuleNotFoundError:
    from tensorflow.lite.python.interpreter import Interpreter, load_delegate


logger = logging.getLogger(__name__)

DETECTOR_KEY = "edgetpu"

# Post-processing parameters (can be adjusted)
CONF_THRESHOLD = 0.7
IOU_THRESHOLD = 0.4
# Frigate only processes up to 20 detections per frame
MAX_DET = 20

class EdgeTpuDetectorConfig(BaseDetectorConfig):
    type: Literal[DETECTOR_KEY]
    device: str = Field(default=None, title="Device Type")


class EdgeTpuTfl(DetectionApi):
    type_key = DETECTOR_KEY

    def __init__(self, detector_config: EdgeTpuDetectorConfig):
        super().__init__(detector_config)
        device_config = {}
        if detector_config.device is not None:
            device_config = {"device": detector_config.device}

        edge_tpu_delegate = None

        try:
            device_type = (
                device_config["device"] if "device" in device_config else "auto"
            )
            logger.info(f"Attempting to load TPU as {device_type}")
            edge_tpu_delegate = load_delegate("libedgetpu.so.1.0", device_config)
            logger.info("TPU found")
            self.interpreter = Interpreter(
                model_path=detector_config.model.path,
                experimental_delegates=[edge_tpu_delegate],
            )
        except ValueError:
            _, ext = os.path.splitext(detector_config.model.path)

            if ext and ext != ".tflite":
                logger.error(
                    "Incorrect model used with EdgeTPU. Only .tflite models can be used with a Coral EdgeTPU."
                )
            else:
                logger.error(
                    "No EdgeTPU was detected. If you do not have a Coral device yet, you must configure CPU detectors."
                )

            raise

        self.interpreter.allocate_tensors()

        self.tensor_input_details = self.interpreter.get_input_details()
        self.tensor_output_details = self.interpreter.get_output_details()
        self.model_type = detector_config.model.model_type

        # Get model input dimensions
        input_shape = self.tensor_input_details[0]["shape"]
        self.height = input_shape[1]
        self.width = input_shape[2]
        logger.info(f"Model input shape: {input_shape}")
        logger.info(f"Model output details: {self.tensor_output_details}")

    def process_yolo_output(self, output_tensor):
        # Get quantization parameters
        output_details = self.tensor_output_details[0]
        scale, zero_point = output_details['quantization']
        
        # Dequantize the output
        output = (output_tensor.astype(np.float32) - zero_point) * scale
        logger.debug(f"Dequantized output shape: {output.shape}")
        logger.debug(f"Dequantized output min/max: {output.min()}/{output.max()}")
        
        # Reshape to [1, 84, 2100]
        predictions = output[0].T  # Transpose to [2100, 84]
        logger.debug(f"Predictions shape after transpose: {predictions.shape}")
        
        # Get boxes and scores
        boxes = predictions[:, :4]  # First 4 values are box coordinates
        scores = predictions[:, 4:]  # Remaining values are class scores
        
        # Normalize scores to [0, 1] range using sigmoid
        scores = 1 / (1 + np.exp(-scores))
        logger.debug(f"Boxes shape: {boxes.shape}, Scores shape: {scores.shape}")
        
        # Get class IDs and confidences
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)
        logger.debug(f"Number of detections before confidence filter: {len(confidences)}")
        logger.debug(f"Confidence range: {confidences.min()}/{confidences.max()}")
        
        # Filter by confidence
        mask = confidences > CONF_THRESHOLD
        boxes = boxes[mask]
        class_ids = class_ids[mask]
        confidences = confidences[mask]
        logger.debug(f"Number of detections after confidence filter: {len(confidences)}")
        
        if len(boxes) == 0:
            logger.debug("No detections after confidence filtering")
            return np.zeros((MAX_DET, 6), np.float32)
        
        # Convert boxes from [x, y, w, h] to [x1, y1, x2, y2]
        boxes_xyxy = np.zeros_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2
        
        # Ensure boxes are within image bounds and have valid dimensions
        boxes_xyxy[:, 0] = np.clip(boxes_xyxy[:, 0], 0, self.width)  # x1
        boxes_xyxy[:, 1] = np.clip(boxes_xyxy[:, 1], 0, self.height)  # y1
        boxes_xyxy[:, 2] = np.clip(boxes_xyxy[:, 2], 0, self.width)  # x2
        boxes_xyxy[:, 3] = np.clip(boxes_xyxy[:, 3], 0, self.height)  # y2
        
        # Ensure minimum box dimensions
        min_box_size = 1.0
        width = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
        height = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]
        
        # Filter out boxes that are too small
        valid_boxes = (width >= min_box_size) & (height >= min_box_size)
        boxes_xyxy = boxes_xyxy[valid_boxes]
        class_ids = class_ids[valid_boxes]
        confidences = confidences[valid_boxes]
        
        if len(boxes_xyxy) == 0:
            logger.debug("No valid boxes after size filtering")
            return np.zeros((MAX_DET, 6), np.float32)
        
        # Normalize coordinates
        boxes_xyxy[:, [0, 2]] /= self.width
        boxes_xyxy[:, [1, 3]] /= self.height
        
        # Apply class-aware NMS
        final_indices = []
        for class_id in np.unique(class_ids):
            class_mask = (class_ids == class_id)
            indices = cv2.dnn.NMSBoxes(
                boxes_xyxy[class_mask], confidences[class_mask], score_threshold=CONF_THRESHOLD, nms_threshold=IOU_THRESHOLD
            )
            if len(indices) > 0:
                # Map back to original indices
                original_indices = np.where(class_mask)[0][indices.flatten()]
                final_indices.extend(original_indices)
        
        # Limit to MAX_DET after NMS across all classes
        final_indices = np.array(final_indices)[:MAX_DET]
        logger.debug(f"Number of detections after NMS: {len(final_indices)}")
        
        # Create detections array
        detections = np.zeros((MAX_DET, 6), np.float32)
        
        if len(final_indices) > 0:
            # Populate detections
            for i, idx in enumerate(final_indices):
                detections[i] = [
                    class_ids[idx],
                    confidences[idx],
                    boxes_xyxy[idx][1],  # y1
                    boxes_xyxy[idx][0],  # x1
                    boxes_xyxy[idx][3],  # y2
                    boxes_xyxy[idx][2],  # x2
                ]
                logger.debug(f"Detection {i}: class={class_ids[idx]}, conf={confidences[idx]:.2f}, box={boxes_xyxy[idx]}")
        
        return detections

    def detect_raw(self, tensor_input):
        self.interpreter.set_tensor(self.tensor_input_details[0]["index"], tensor_input)
        self.interpreter.invoke()

        if self.model_type == ModelTypeEnum.yologeneric:
            # For YOLO models, we expect a single output tensor
            output_tensor = self.interpreter.tensor(self.tensor_output_details[0]["index"])()
            logger.debug(f"YOLO output tensor shape: {output_tensor.shape}")
            return self.process_yolo_output(output_tensor)
        else:
            # For other models (like SSD), use the existing detection logic
            try:
                boxes = self.interpreter.tensor(self.tensor_output_details[0]["index"])()[0]
                class_ids = self.interpreter.tensor(self.tensor_output_details[1]["index"])()[0]
                scores = self.interpreter.tensor(self.tensor_output_details[2]["index"])()[0]
                count = int(
                    self.interpreter.tensor(self.tensor_output_details[3]["index"])()[0]
                )

                detections = np.zeros((20, 6), np.float32)

                for i in range(count):
                    if scores[i] < 0.4 or i == 20:
                        break
                    detections[i] = [
                        class_ids[i],
                        float(scores[i]),
                        boxes[i][0],
                        boxes[i][1],
                        boxes[i][2],
                        boxes[i][3],
                    ]

                return detections
            except Exception as e:
                logger.error(f"Error processing model output: {str(e)}")
                logger.error(f"Model output details: {self.tensor_output_details}")
                raise
