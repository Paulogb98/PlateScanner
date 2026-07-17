import numpy as np
import cv2
import os
import torch
import traceback
from yolov5.models.yolo import DetectionModel


class YOLOv5Inference:
    def __init__(self, model_path, conf=0.25, iou=0.45, agnostic=False, multi_label=False, max_det=1):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        checkpoint = torch.load(model_path, weights_only=False, map_location=self.device)

        self.model = DetectionModel(checkpoint['model'].yaml)
        self.model.load_state_dict(checkpoint['model'].state_dict())
        self.model.to(self.device).eval()

        self.conf = conf
        self.iou = iou
        self.agnostic = agnostic
        self.multi_label = multi_label
        self.max_det = max_det


    def infer(self, img_url, size=640):
        """Runs inference on the specified image."""
        try:
            img = cv2.imread(img_url)
            if img is None:
                raise ValueError(f"Could not load image: {img_url}")

            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            h, w = img.shape[:2]
            r = size / max(h, w)
            if r != 1:
                img_resized = cv2.resize(img, (int(w * r), int(h * r)), interpolation=cv2.INTER_LINEAR)
            else:
                img_resized = img

            h_new, w_new = img_resized.shape[:2]
            top = bottom = (size - h_new) // 2
            left = right = (size - w_new) // 2
            img_padded = cv2.copyMakeBorder(
                img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
            )

            img_tensor = torch.from_numpy(img_padded).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(self.device)

            with torch.no_grad():
                pred = self.model(img_tensor)[0]

            pred = self.non_max_suppression(
                pred, conf_thres=self.conf, iou_thres=self.iou,
                agnostic=self.agnostic, multi_label=self.multi_label, max_det=self.max_det
            )

            class Results:
                def __init__(self, pred, img):
                    self.pred = pred
                    self.ims = [img]

            results = Results(pred, img_padded)
            return results

        except Exception as e:
            print(f"Error processing image: {e}")
            traceback.print_exc()
            return None


    def non_max_suppression(self, prediction, conf_thres=0.25, iou_thres=0.45, agnostic=False, multi_label=False, max_det=300):
        """Applies Non-Maximum Suppression (NMS) to inference results."""
        from yolov5.utils.general import non_max_suppression as nms
        return nms(prediction, conf_thres, iou_thres, agnostic=agnostic, multi_label=multi_label, max_det=max_det)


    def process_results(self, results, img_name, output_dir='results', margin_factor=0.2):
        """Processes inference results, draws boxes, and saves images."""
        if results is None:
            return

        predictions = results.pred[0]
        if predictions.shape[0] == 0:
            print("No detections found after NMS.")
            return

        boxes = predictions[:, :4]

        detected_dir = os.path.join(output_dir, 'detected')
        cropped_dir = os.path.join(output_dir, 'cropped')
        os.makedirs(detected_dir, exist_ok=True)
        os.makedirs(cropped_dir, exist_ok=True)

        img = results.ims[0].copy()
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)

        for box in boxes:
            x1, y1, x2, y2 = map(int, box)

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            detected_path = os.path.join(detected_dir, f'{os.path.splitext(img_name)[0]}_detected{os.path.splitext(img_name)[1]}')
            cv2.imwrite(detected_path, img)

            cropped_image = img[y1:y2, x1:x2]
            cropped_path = os.path.join(cropped_dir, f'{os.path.splitext(img_name)[0]}_cropped{os.path.splitext(img_name)[1]}')
            cv2.imwrite(cropped_path, cropped_image)


    def process_directory(self, input_dir, output_dir):
        """Processes all images in a directory."""
        os.makedirs(output_dir, exist_ok=True)

        for img_name in os.listdir(input_dir):
            img_path = os.path.join(input_dir, img_name)
            if os.path.isfile(img_path) and os.path.splitext(img_path)[1].lower() in ['.png', '.jpg', '.jpeg']:
                results = self.infer(img_path)
                self.process_results(results, img_name, output_dir)
