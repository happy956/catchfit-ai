from huggingface_hub import snapshot_download
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
import numpy as np
import torch
import yaml
import cv2

labels = {
    0: 'Background',
    1: 'Apparel',
    2: 'Face_Neck',
    3: 'Hair',
    4: 'Left_Foot',
    5: 'Left_Hand',
    6: 'Left_Lower_Arm',
    7: 'Left_Lower_Leg',
    8: 'Left_Shoe',
    9: 'Left_Sock',
    10: 'Left_Upper_Arm',
    11: 'Left_Upper_Leg',
    12: 'Lower_Clothing',
    13: 'Right_Foot',
    14: 'Right_Hand',
    15: 'Right_Lower_Arm',
    16: 'Right_Lower_Leg',
    17: 'Right_Shoe',
    18: 'Right_Sock',
    19: 'Right_Upper_Arm',
    20: 'Right_Upper_Leg',
    21: 'Torso',
    22: 'Upper_Clothing',
    23: 'Lower_Lip',
    24: 'Upper_Lip',
    25: 'Lower_Teeth',
    26: 'Upper_Teeth',
    27: 'Tongue'
}

ORIGINAL_GOLIATH_CLASSES = (
    "Background",
    "Apparel",
    "Chair",
    "Eyeglass_Frame",
    "Eyeglass_Lenses",
    "Face_Neck",
    "Hair",
    "Headset",
    "Left_Foot",
    "Left_Hand",
    "Left_Lower_Arm",
    "Left_Lower_Leg",
    "Left_Shoe",
    "Left_Sock",
    "Left_Upper_Arm",
    "Left_Upper_Leg",
    "Lower_Clothing",
    "Lower_Spandex",
    "Right_Foot",
    "Right_Hand",
    "Right_Lower_Arm",
    "Right_Lower_Leg",
    "Right_Shoe",
    "Right_Sock",
    "Right_Upper_Arm",
    "Right_Upper_Leg",
    "Torso",
    "Upper_Clothing",
    "Visible_Badge",
    "Lower_Lip",
    "Upper_Lip",
    "Lower_Teeth",
    "Upper_Teeth",
    "Tongue",
)

GOLIATH_CLASSES = list(range(len(ORIGINAL_GOLIATH_CLASSES)))

ORIGINAL_GOLIATH_PALETTE = [
    [50, 50, 50],
    [255, 218, 0],
    [102, 204, 0],
    [14, 0, 204],
    [0, 204, 160],
    [128, 200, 255],
    [255, 0, 109],
    [0, 255, 36],
    [189, 0, 204],
    [255, 0, 218],
    [0, 160, 204],
    [0, 255, 145],
    [204, 0, 131],
    [182, 0, 255],
    [255, 109, 0],
    [0, 255, 255],
    [72, 0, 255],
    [204, 43, 0],
    [204, 131, 0],
    [255, 0, 0],
    [72, 255, 0],
    [189, 204, 0],
    [182, 255, 0],
    [102, 0, 204],
    [32, 72, 204],
    [0, 145, 255],
    [14, 204, 0],
    [0, 128, 72],
    [204, 0, 43],
    [235, 205, 119],
    [115, 227, 112],
    [157, 113, 143],
    [132, 93, 50],
    [82, 21, 114],
]

class Segmentator:
    def __init__(self, args):
        self.args = args
        self.device = self.args['device']
        self.model_version = self.args['segmentator_model_version']
        self.model_weight = self.args['segmentator_model_weight']
        snapshot_download(repo_id=f'facebook/sapiens-seg-{self.model_version}-torchscript', local_dir='weights')
        self.model = torch.jit.load(f'weights/{self.model_weight}').to(device=self.device)
        self.model.eval()
        
        self.transformer = transforms.Compose([
            transforms.Resize((1024, 768)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[123.5/255, 116.5/255, 103.5/255], std=[58.5/255, 57.0/255, 57.5/255]),
        ])
    
    @torch.inference_mode()
    def get_segment(self, img):
        H, W, C = img.shape
        img = Image.fromarray(img)
        img = self.transformer(img).unsqueeze(0).to(device=self.device)
        output = self.model(img)
        output = F.interpolate(output, size=(H, W), mode='bilinear', align_corners=False)
        _, preds = torch.max(output, 1)
        mask = preds.squeeze(0).cpu().numpy()
        return mask
    
    def visualize(self, img, segment, alpha=0.5):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img_rgb)
        img_np = np.array(img)
        
        ids = np.unique(segment)[::-1]
        labels = np.array([i for i in ids if i in GOLIATH_CLASSES], dtype=np.int64)
        colors = [ORIGINAL_GOLIATH_PALETTE[label] for label in labels]
        
        overlay = np.zeros((*segment.shape, 3), dtype=np.uint8)
        for label, color in zip(labels, colors):
            overlay[segment == label, :] = color
        
        result = np.uint8(img_np * (1 - alpha) + overlay * alpha)
        return result

if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        args = yaml.safe_load(f)
    
    img = cv2.imread('data/dataset1/IMG_6849.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    model = Segmentator(args)
    segment = model.get_segment(img)
    
    result = model.visualize(img, segment)
    cv2.imwrite('results/segmentation_test.jpg', result)