import torch
import cv2
import argparse
import numpy as np

from Models import Unet

def load_model(weight_path, device):
    """모델을 로드하고 초기화합니다."""
    n_classes = 2
    model = Unet(encoder='resnet34', pre_weight='imagenet', num_classes=n_classes).to(device)
    model.model.load_state_dict(torch.load(weight_path, map_location=torch.device(device)))
    model.eval()
    return model

def preprocess_image(img_path):
    """이미지를 로드하고 전처리합니다."""
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    img_input = img / 255.
    img_input = img_input.transpose([2, 0, 1])
    img_input = torch.tensor(img_input).float()
    return img_input, img

def predict(model, img_input, device):
    """모델을 사용하여 이미지에 대한 추론을 실행합니다."""
    img_input = img_input.unsqueeze(0).to(device)
    output = model(img_input)
    img_output = torch.argmax(output, dim=1).detach().cpu().numpy().squeeze()
    return img_output

def create_overlay_and_save(img, img_output, output_path):
    """추론 결과를 원본 이미지 위에 오버레이하고 저장합니다."""
    overlay = np.zeros_like(img)
    overlay[img_output == 1] = [0, 0, 255]
    alpha = 0.25
    result_img = cv2.addWeighted(overlay, alpha, img.astype(np.uint8), 1 - alpha, 0)
    cv2.imwrite(output_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))

def main():
    parser = argparse.ArgumentParser(description='Process some images.')
    parser.add_argument('img_path', type=str, help='Path to the input image')
    args = parser.parse_args()

    img_path = args.img_path 
    weight_path = './0310_crushed.pt'
    device = 'cpu'

    model = load_model(weight_path, device)
    print('Loaded pretrained model!')

    img_input, original_img = preprocess_image(img_path)
    img_input = img_input.to(device)

    img_output = predict(model, img_input, device)
    overlayed_output_path = './overlayed_output_image.jpg'
    create_overlay_and_save(original_img, img_output, overlayed_output_path)

    print(f'Saved overlayed output image to {overlayed_output_path}')

if __name__ == "__main__":
    main()

# wip parser 에 device 설정.