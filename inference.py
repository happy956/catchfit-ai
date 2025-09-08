from agent import Agent
import numpy as np
import torch
import yaml
import cv2

if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        args = yaml.safe_load(f)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args['device'] = device
    
    img = cv2.imread('data/dataset1/IMG_6849.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img1 = cv2.imread('data/dataset1/IMG_6850.jpg')
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    
    agent = Agent(args)
    
    from time import time
    
    s_t1 = time()
    agent.set_img(img)
    s_t2 = time()
    
    length1 = agent.calculate()
    s_t3 = time()
    
    agent.set_img(img1)
    length2 = agent.calculate()
    agent.set_img(img)
    
    s_t4 = time()
    outline_error = agent.get_outline_error(img1)
    keypoint_error = agent.get_keypoints_error(img1)
    s_t5 = time()
    
    print(length1)
    print(length2)
    print(outline_error)
    print(keypoint_error)
    print(s_t2 - s_t1, s_t3 - s_t2, s_t5 - s_t4)