from ultralytics import YOLO
import numpy as np
import torch
import yaml
import cv2

class BBOXDetector:
    def __init__(self, args):
        self.args = args
        self.device = self.args['device']
        self.model_weight = self.args['bbox_model_weight']
        self.model = YOLO(f'weights/{self.model_weight}').to(device=self.device)
        self.bbox_classes = self.args['bbox_classes']
        self.model.eval()
    
    def get_bbox(self, img):
        results = self.model(img, classes=self.bbox_classes)
        
        max_bbox = None
        max_area = -np.inf
        for result in results[0]:
            _, _, w, h = result.boxes.xywhn.cpu().detach().numpy()[0]
            area = w * h
            
            if area > max_area:
                max_area = area
                max_bbox = result.boxes.xyxy.cpu().detach().numpy()[0]
            
            bbox = max_bbox.astype('int')
            return bbox

if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        args = yaml.safe_load(f)
    
    img_path = 'data/dataset1/IMG_6849.jpg'
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W, C = img.shape
    
    model = BBOXDetector(args)
    bbox = model.get_bbox(img)
    print(bbox)