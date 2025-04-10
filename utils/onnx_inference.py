import cv2
import numpy as np
import onnxruntime as ort
from .helpers import CLASS_NAMES, COLORS, preprocess, postprocess

class YOLOv11:
    def __init__(self, onnx_path, conf_thres=0.5, iou_thres=0.5):
        self.session = ort.InferenceSession(onnx_path)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # Verify input type
        input_type = self.session.get_inputs()[0].type
        assert "float" in input_type, f"Model expects {input_type}"

    def detect(self, image):
        orig_h, orig_w = image.shape[:2]
        blob = preprocess(image)
        outputs = self.session.run([self.output_name], {self.input_name: blob})
        boxes, scores, class_ids = postprocess(outputs, self.conf_thres, self.iou_thres)
        
        results = []
        for box, score, class_id in zip(boxes, scores, class_ids):
            x, y, w, h = box
            x1 = int(x * orig_w / 640)
            y1 = int(y * orig_h / 640)
            x2 = int((x + w) * orig_w / 640)
            y2 = int((y + h) * orig_h / 640)
            
            results.append({
                'class': CLASS_NAMES[class_id],
                'confidence': score,
                'box': [x1, y1, x2, y2]
            })
        return results

    def draw_detections(self, image, detections):
        for det in detections:
            x1, y1, x2, y2 = det['box']
            color = COLORS[CLASS_NAMES.index(det['class'])]
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            label = f"{det['class']}: {det['confidence']:.2f}"
            cv2.putText(image, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return image