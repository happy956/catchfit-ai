from sam2.sam2_image_predictor import SAM2ImagePredictor
from .bbox_detector import BBOXDetector
import numpy as np
import torch
import yaml
import cv2

class OutLiner:
    def __init__(self, args):
        self.args = args
        self.device = self.args['device']
        self.model_weight = self.args['outline_model_weight']
        self.model = SAM2ImagePredictor.from_pretrained(self.model_weight)
    
    @torch.inference_mode()
    @torch.autocast('cuda', dtype=torch.bfloat16)
    def get_outline_mask(self, img, bbox=None):
        self.model.set_image(img)
        if bbox is not None:
            mask, _, _ = self.model.predict(box=bbox, multimask_output=False)
        else:
            mask, _, _ = self.model.predict(multimask_output=False)
            mask = mask.astype(np.uint8)
        
        mask[mask == 1] = 255
        if len(mask) != 1:
            raise
        else:
            return mask[0]

if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        args = yaml.safe_load(f)
    
    img_path = 'data/dataset1/IMG_6849.jpg'
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W, C = img.shape
    
    bbox_model = BBOXDetector(args)
    model = OutLiner(args)
    
    bbox = bbox_model.get_bbox(img)
    outline = model.get_outline(img, bbox)
    cv2.imwrite('results/outline_test.jpg', outline.reshape(H, W, 1))