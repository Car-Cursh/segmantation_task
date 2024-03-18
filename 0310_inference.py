
import torch
import cv2
import argparse  # argparse 모듈 추가
import numpy as np

from Models import Unet

parser = argparse.ArgumentParser(description='Process some images.')
parser.add_argument('img_path', type=str, help='Path to the input image')
args = parser.parse_args()

img_path = args.img_path  # 커맨드라인에서 받은 이미지 경로
print(img_path)
weight_path = './0310_crushed.pt'
n_classes = 2

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
model = Unet(encoder='resnet34', pre_weight='imagenet', num_classes=n_classes).to(device)
model.model.load_state_dict(torch.load(weight_path, map_location=torch.device(device)))
model.eval()

print('Loaded pretrained model!')

img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (256, 256))
img_input = img / 255.
img_input = img_input.transpose([2, 0, 1])
img_input = torch.tensor(img_input).float().to(device)

img_input = img_input.unsqueeze(0)
output = model(img_input)

# 추론 마스크 만들기
img_output = torch.argmax(output, dim=1).detach().cpu().numpy()
img_output = img_output.squeeze()  

overlay = np.zeros_like(img)

overlay[img_output == 1] = [0, 0, 255]


alpha = 0.25
result_img = cv2.addWeighted(overlay, alpha, img.astype(np.uint8), 1 - alpha, 0)

# 이미지 저장하기
overlayed_output_path = './overlayed_output_image.jpg'
cv2.imwrite(overlayed_output_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))  # 저장 시 BGR로 변환

print(f'Saved overlayed output image to {overlayed_output_path}')