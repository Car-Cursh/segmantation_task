import torch
import cv2
import matplotlib.pyplot as plt
import argparse
import numpy as np

from Models import Unet

parser = argparse.ArgumentParser(description='Process some images.')
parser.add_argument('img_path', type=str, help='Path to the input image')
args = parser.parse_args()

img_path = args.img_path
weight_path = './40_model.pth'
n_classes = 1  # 이진 분류를 멀티 클래스처럼 하지 않고 회귀느낌으로다가.

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Unet(encoder='resnet34', pre_weight=None, num_classes=n_classes).to(device)
model.load_state_dict(torch.load(weight_path, map_location=torch.device(device)))
model.eval()

img = cv2.imread(img_path)
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

overlayed_output_path = './overlayed_output_image.jpg'
cv2.imwrite(overlayed_output_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))

print(f'Saved overlayed output image to {overlayed_output_path}')
