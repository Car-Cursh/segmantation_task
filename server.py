from typing import Annotated

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

import numpy as np
import cv2
from io import BytesIO
import torch
from Models import Unet

device = 'cuda' if torch.cuda.is_available() else 'cpu'

app = FastAPI()

#CSS 설정 불러오기
@app.get("/styleIndex.css")
async def main0():
    return FileResponse('styleIndex.css') 
@app.get("/styleMenuBar.css")
async def main0():
    return FileResponse('styleMenuBar.css') 
@app.get("/uploadButton.css")
async def main0():
    return FileResponse('uploadButton.css') 


#최초 서버 실행시 첫화면 -> index.html을 호출 = 메인 홈페이지 출력
@app.get("/")
async def main1():
    return FileResponse('index.html') #view/index.html에서 수정
#메뉴바에서 홈을 눌렀을때 홈페이지 출력
@app.get("/index.html")
async def main1():
    return FileResponse('index.html') 
#uploadimages.html을 호출 = 이미지업로드페이지 출력
@app.get("/uploadImages.html")
async def main2():
    return FileResponse('uploadImages.html') 


#업로드된 파일을 흑백사진으로 변환 후 이미지를 반환
@app.post("/upload")
async def create_upload_file(file: UploadFile):
    contents = await file.read() #contents에 업로드된 이미지 저장
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    ############
    weight_path = './70_model.pth'
    n_classes = 1  # 이진 분류를 멀티 클래스처럼 하지 않고 회귀느낌으로다가.

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Unet(encoder='resnet34', pre_weight=None, num_classes=n_classes).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=torch.device(device)))
    model.eval()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    img_input = img / 255.
    img_input = img_input.transpose([2, 0, 1])
    img_input = torch.tensor(img_input).float().to(device)

    img_input = img_input.unsqueeze(0)
    output = model(img_input)

    # 시그모이드로 바꿈.
    probabilities = torch.sigmoid(output)
    img_output = (probabilities > 0.5).float().detach().cpu().numpy()
    img_output = img_output.squeeze()

    overlay = np.zeros_like(img)
    overlay[img_output == 1] = [0, 0, 255]

    alpha = 0.25
    result_img = cv2.addWeighted(overlay, alpha, img.astype(np.uint8), 1 - alpha, 0)

    # Convert back to bytes
    _, encoded_img = cv2.imencode('.PNG', result_img)
    byte_io = BytesIO(encoded_img.tobytes())
    return StreamingResponse(byte_io, media_type="image/png")


