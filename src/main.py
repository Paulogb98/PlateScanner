import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from classes.detector import YOLOv5Inference
from classes.recognizer import ONNXPlateRecognizer


def main():
    if not os.path.exists(model_path_detection):
        print(f"Error: File {model_path_detection} not found.")
        sys.exit(1)

    print("Initializing detection model...")
    yolo_detector = YOLOv5Inference(model_path_detection)

    print(f"Processing images from directory: {input_images_dir}")
    yolo_detector.process_directory(input_images_dir, output_images_dir)

    if not os.path.exists(model_path_recognition):
        print(f"Error: File {model_path_recognition} not found.")
        sys.exit(1)

    print("Initializing recognition model...")
    recognizer = ONNXPlateRecognizer(model_path_recognition, config_path_recognition)

    print(f"Processing cropped images from directory: {cropped_images_dir}")
    recognizer.process_cropped_images(cropped_images_dir, results_dir)

    print(f"Results saved to: {results_dir}")


model_path_detection = '/app/models/detector/yolo_detector_model.pt'
config_path_recognition = "/app/models/recognizer/license_plates_ocr_config.yaml"
model_path_recognition = "/app/models/recognizer/license_plates_ocr_model.onnx"

input_images_dir = '/app/data/raw'
output_images_dir = '/app/data/processed'
cropped_images_dir = '/app/data/processed/cropped'
results_dir = '/app/data/results'

if __name__ == "__main__":
    main()
