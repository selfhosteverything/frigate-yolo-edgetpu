# frigate-yolo-edgetpu
Patch Frigate edgetpu_tfl.py to use YOLOv8 with edgetpu
Blog post: https://selfhosteverything.xyz/yolov8-coral-edgetpu-frigate/

# Using it
Replace `/frigate/detectors/plugins/edgetpu_tfl.py`

Added post processing function for yolov8 tensor output `process_yolov8_output`

Frigate config to use it:
```
detectors:
  coral:
    type: edgetpu
    device: usb

model:
  path: /config/custom_models/yolov8n_300e_edgetpu.tflite
  labelmap_path: /config/custom_models/3labels.txt
  model_type: yolo-generic
  width: 320
  height: 320
```

