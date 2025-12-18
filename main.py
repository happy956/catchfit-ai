from fastapi import FastAPI, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from agent import Agent
from modules import *
import numpy as np
import uvicorn
import base64
import torch
import uuid
import yaml
import cv2

app = FastAPI()

jobs = {}

origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:3000/*",
    "http://localhost:3001",
    "http://localhost:3001/*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

with open('config.yaml', 'r') as f:
    args = yaml.safe_load(f)
args['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

@app.get('/')
def home():
    return {'result': "CatchFit API Server"}

@app.post('/outline')
async def get_outline(file: UploadFile):
    content = await file.read()
    encoded_img = np.frombuffer(content, dtype=np.uint8)
    img = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
    agent = Agent(args)
    outline = agent.get_outline(img)
    return {'result': outline}

@app.post('/keypoints')
async def get_keypoints(file: UploadFile):
    content = await file.read()
    encoded_img = np.frombuffer(content, dtype=np.uint8)
    img = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
    agent = Agent(args)
    _, keypoints = agent.get_keypoints(img)
    return {'result': keypoints}

@app.post('/ratio')
async def get_ratio(file1: UploadFile, file2: UploadFile):
    content1 = await file1.read()
    content2 = await file2.read()
    encoded_img1 = np.frombuffer(content1, np.uint8)
    img1 = cv2.imdecode(encoded_img1, cv2.IMREAD_COLOR)
    encoded_img2 = np.frombuffer(content2, np.uint8)
    img2 = cv2.imdecode(encoded_img2, cv2.IMREAD_COLOR)
    
    agent = Agent(args)
    agent.set_img(img1)
    length1 = agent.calculate()

    agent.set_img(img2)
    length2 = agent.calculate()

    length1, length2 = np.array(length1), np.array(length2)
    length1_ = np.where(length1 == 0, 1, length1)
    ratio = (length2 - length1) / (length1_) * 100
    return {'result': ratio}

@app.post('/error')
async def get_error(file1: UploadFile, file2: UploadFile):
    content1 = await file1.read()
    img1 = cv2.imdecode(np.frombuffer(content1, np.uint8), cv2.IMREAD_COLOR)
    content2 = await file2.read()
    img2 = cv2.imdecode(np.frombuffer(content2, np.uint8), cv2.IMREAD_COLOR)

    agent = Agent(args)
    agent.set_img(img1)
    outline_error = agent.get_outline_error(img2)
    keypoints_error = agent.get_keypoints_error(img2)
    return {'result': [outline_error, keypoints_error]}

def object_resize(img1, img2, agent):
    _, keypoints_img1 = agent.get_keypoints(img1)
    _, keypoints_img2 = agent.get_keypoints(img2)
    
    main_keypoints_img1 = keypoints_img1[[5, 6, 7, 8]]
    main_keypoints_img2 = keypoints_img2[[5, 6, 7, 8]]
    
    if -1 in main_keypoints_img1 or -1 in main_keypoints_img2:
        return img2
    
    resized_img2, _, _ = affine_align_shoulders_elbows(img2, main_keypoints_img1, main_keypoints_img2)
    return resized_img2

def process(job_id: str, content1: bytes, content2: bytes):
    jobs[job_id]['progress'] += 5
    
    img1 = cv2.imdecode(np.frombuffer(content1, np.uint8), cv2.IMREAD_COLOR)
    img2 = cv2.imdecode(np.frombuffer(content2, np.uint8), cv2.IMREAD_COLOR)
    
    _, buffer = cv2.imencode('.png', img2)
    jobs[job_id]['result']['img'] = base64.b64encode(buffer).decode('utf-8')
    
    H, W, _ = img2.shape
    
    agent = Agent(args)
    
    _, origin_img1_keypoints = agent.get_keypoints(img1)
    resized_img1 = resize_to_match_letterbox(img1, (H, W))
    
    resized_img1 = object_resize(img2, resized_img1, agent)
    
    agent.set_img(resized_img1)
    mask = np.where(origin_img1_keypoints == -1)
    agent.keypoints[mask] = -1
    img1_calc_data = agent.calculate()
    length1 = img1_calc_data['length']
    points1 = img1_calc_data['points']
    jobs[job_id]['progress'] += 20
    jobs[job_id]['progress'] += 15
    
    outline = agent.get_outline(resized_img1)
    jobs[job_id]['result']['outline'] = outline
    
    _, keypoints = agent.get_keypoints(img2)
    jobs[job_id]['progress'] += 25
    jobs[job_id]['result']['keypoints'] = keypoints.tolist()
    
    
    agent.set_img(img2)
    img2_calc_data = agent.calculate()
    length2 = img2_calc_data['length']
    points2 = img2_calc_data['points']
    jobs[job_id]['progress'] += 25
    
    converted_points2 = []
    for p1, p2 in zip(points1, points2):
        if p1 is None or p2 is None:
            converted_points2.append(None)
        else:
            x, y = p2
            converted_points2.append((float(x), float(y)))
    jobs[job_id]['result']['label_points'] = converted_points2
    
    length1, length2 = np.array(length1), np.array(length2)
    length1_ = np.where(length1 == 0, 1, length1)
    ratio = (length2 - length1) / (length1_) * 100
    jobs[job_id]['progress'] += 10
    jobs[job_id]['result']['ratio'] = ratio.tolist()
    jobs[job_id]['progress'] = 100
    jobs[job_id]['status'] = 'done'

# ---- analysis (백그라운드 태스크로 실행) ----
@app.post('/analysis')
async def analysis(file1: UploadFile, file2: UploadFile, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {'progress': 0, 'status': 'running', 'result': {'img': None, 'outline': None, 'keypoints': None, 'ratio': None, 'label_points': None}}

    content1 = await file1.read()
    content2 = await file2.read()
    
    # ✅ BackgroundTasks로 등록
    background_tasks.add_task(process, job_id, content1, content2)
    return {"job_id": job_id}

@app.get("/progress/{job_id}")
async def get_progress(job_id: str):
    if job_id not in jobs:
        return JSONResponse(staus_code=404, content={"error": "Job not found"})
    return {"progress": jobs[job_id]["progress"], "status": jobs[job_id]["status"]}

@app.get("/result/{job_id}")
async def get_result(job_id: str):
    if job_id not in jobs:
        return JSONResponse(status_code=404, content={"error": "Job not found"})
    status = jobs[job_id]['status']
    res = True if status == 'done' else False
    return {'status': status, 'data': jobs[job_id]['result'], 'result': res}

@app.delete("/result/{job_id}")
async def delete_job(job_id: str):
    if job_id not in jobs:
        return JSONResponse(status_code=404, content={"error": "job ID is not found"})
    del jobs[job_id]
    return {"message": f"{job_id} deleted successfully"}

if __name__ == '__main__':
    uvicorn.run('main:app', host='0.0.0.0', port=8080, reload=True)