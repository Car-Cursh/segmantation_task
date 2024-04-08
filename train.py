import glob
import json
import os
from PIL import Image, ImageDraw
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import torch
from torch import nn, optim
import segmentation_models_pytorch as smp
from torchmetrics import BinaryJaccardIndex
from Models import Unet

class CarDamageDataset(Dataset):
    def __init__(self, json_dir, img_dir, transform=None):
        self.json_files = glob.glob(os.path.join(json_dir, "*.json"))
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.json_files)

    def __getitem__(self, idx):
        json_path = self.json_files[idx]
        with open(json_path, 'r') as j:
            annotation = json.load(j)

        img_file_name = os.path.splitext(os.path.basename(json_path))[0] + '.jpg' # GT와 같은 파일명 찾기
        img_path = os.path.join(self.img_dir, img_file_name)

        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"WUT??? Image file not found : {img_path}")
            raise

        mask = Image.new("L", (image.width, image.height), 0)
        draw = ImageDraw.Draw(mask)
        for seg in annotation['annotations'][0]['segmentation']:
            # GT 구조 = [[[x1, y1], [x2, y2], ..., [xn, yn]]]
            polygon = [(point[0], point[1]) for point in seg[0]]
            draw.polygon(polygon, outline=255, fill=255)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
            mask = (mask>0).float()

        return image, mask
# 변환 정의
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# 데이터셋 및 데이터 로더 초기화
dataset = CarDamageDataset(json_dir='./car_data/all/json',
                           img_dir='./car_data/all/img',
                           transform=transform)
data_loader = DataLoader(dataset, batch_size=128, shuffle=True)

val_dataset =CarDamageDataset(json_dir = './car_data/val/json',
                            img_dir = './car_data/val/img',
                            transform = transform)
val_data_loader = DataLoader(val_dataset,batch_size = 128,shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 1  # Not background and damaged part, Just percentage of damaged pixel.
model = Unet(num_classes=num_classes, encoder='resnet34', pre_weight='imagenet').to(device)

pos_weight = torch.tensor([10]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # For binary classification, adjust if necessary
optimizer = optim.Adam(model.parameters(), lr=0.001)

# BinaryJaccardIndex 인스턴스 생성
binary_jaccard_index = BinaryJaccardIndex().to(device)

num_epochs = 100
save_dir = './save'

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    # Validation Loop
    if epoch % 10 == 9:
        total_jaccard_index = 0.0
        count = 0
        model.eval()
        with torch.no_grad():
            for idx, (images, masks) in val_data_loader:
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                preds = torch.sigmoid(outputs) > 0.5
                preds = preds.float()  # BinaryJaccardIndex를 위해 preds를 float로 변경

                # BinaryJaccardIndex 계산
                score = binary_jaccard_index(preds, masks.float())  # masks도 float으로 변경
                total_jaccard_index += score.item()  # score.item()으로 스칼라 값 접근
                count += 1

            mean_jaccard_index = total_jaccard_index / count
            print(f"Epoch {epoch+1}, mIOU: {mean_jaccard_index:.4f}")
            torch.save(model.state_dict(), os.path.join(save_dir, f'{epoch+1}_model.pth'))

    for images, masks in data_loader:
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(data_loader.dataset)
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}')

print('Finished Training')