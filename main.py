from fastapi import FastAPI, UploadFile
from agent import Agent
import numpy as np
import torch
import yaml
import cv2

app = FastAPI()

@app.get('/')
def home():
    return {'result': "CatchFit API Server"}

@app.post('/outline')
async def get_outline(file: UploadFile):
    content = await file.read()
    encoded_img = np.fromstring(content, dtype=np.uint8)
    img = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
    outline = agent.get_outline(img)
    return {'result': outline}

@app.post('/keypoints')
async def get_keypoints(file: UploadFile):
    content = await file.read()
    encoded_img = np.fromstring(content, dtype=np.uint8)
    img = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
    _, keypoints = agent.get_keypoints(img)
    return {'result': keypoints.tolist()}

@app.post('/ratio')
async def get_ratio(file1: UploadFile, file2: UploadFile):
    content1 = await file1.read()
    encoded_img1 = np.fromstring(content1, dtype=np.uint8)
    img1 = cv2.imdecode(encoded_img1, cv2.IMREAD_COLOR)
    
    content2 = await file2.read()
    encoded_img2 = np.fromstring(content2, dtype=np.uint8)
    img2 = cv2.imdecode(encoded_img2, cv2.IMREAD_COLOR)
    
    agent.set_img(img1)
    length1 = agent.calculate()
    
    agent.set_img(img2)
    length2 = agent.calculate()
    
    length1, length2 = np.array(length1), np.array(length2)
    length1_ = np.where(length1 == 0, 1, length1)
    ratio = (length2 - length1) / (length1_) * 100
    return {'result': ratio.tolist()}

@app.post('/error')
async def get_error(file1: UploadFile, file2: UploadFile):
    content1 = await file1.read()
    encoded_img1 = np.fromstring(content1, dtype=np.uint8)
    img1 = cv2.imdecode(encoded_img1, cv2.IMREAD_COLOR)
    
    content2 = await file2.read()
    encoded_img2 = np.fromstring(content2, dtype=np.uint8)
    img2 = cv2.imdecode(encoded_img2, cv2.IMREAD_COLOR)
    
    agent.set_img(img1)
    outline_error = agent.get_outline_error(img2)
    keypoints_error = agent.get_keypoints_error(img2)
    return {'result': [outline_error, keypoints_error]}

with open('config.yaml', 'r') as f:
    args = yaml.safe_load(f)
args['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

agent = Agent(args)