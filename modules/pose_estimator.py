from ultralytics import YOLO
import numpy as np
import torch
import yaml
import cv2

class PoseEstimator:
    def __init__(self, args):
        self.args = args
        self.device = self.args['device']
        self.model_weight = self.args['pose_estimator_model_weight']
        self.model = YOLO(f'weights/{self.model_weight}').to(device=self.device)
        self.model.eval()
    
    def get_keypoints(self, img):
        output = self.model(img)
        result = output[0]
        
        max_area = -np.inf
        for box, keypoint in zip(result.boxes, result.keypoints):
            _, _, w, h = box.xywhn[0]
            area = w * h
            if area > max_area:
                max_area = area
                max_keypoint = keypoint.xy[0].cpu().detach().numpy()
                max_xyxy = box.xyxy[0].cpu().detach().numpy()
        return max_xyxy, max_keypoint

if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        args = yaml.safe_load(f)
    args['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    img = cv2.imread('data/dataset1/IMG_6849.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    model = PoseEstimator(args)
    xyxy, keypoint = model.get_keypoints(img)
    keypoint = keypoint.astype('int')
    
    result = img.copy()
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    for x, y in keypoint:
        cv2.circle(result, (x, y), 10, (0, 255, 0), -1, cv2.LINE_AA)
    cv2.imwrite('results/pose_estimation_test.jpg', result)