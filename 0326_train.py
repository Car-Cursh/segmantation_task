import glob
import json
import os
from PIL import Image, ImageDraw
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchmetrics import JaccardIndex

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
            draw.polygon(polygon, outline=1, fill=1)
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
            if mask.size(0) > 1:  # 여러 채널을 가진 경우 첫 번째 채널만 사용
                mask = mask[0].unsqueeze(0)
        
        return image, mask

# 변환 정의
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# 데이터셋 및 데이터 로더 초기화
dataset = CarDamageDataset(json_dir='../../car_data/all/json',
                           img_dir='../../car_data/all/img',
                           transform=transform)
data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

import torch
from torch import nn, optim
import segmentation_models_pytorch as smp


class Unet(nn.Module):
    def __init__(self, num_classes,encoder,pre_weight):
        super().__init__()
        self.model = smp.Unet( classes = num_classes,
                              encoder_name=encoder,
                              encoder_weights=pre_weight,
                              in_channels=3)
    
    def forward(self, x):
        y = self.model(x)
        encoder_weights = "imagenet"
        return y

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 1  # Not background and damaged part , Just percentage of damaged pixel. 
model = Unet(num_classes=num_classes, encoder='resnet34', pre_weight='imagenet').to(device)

criterion = nn.BCEWithLogitsLoss()  # For binary classification, adjust if necessary
optimizer = optim.Adam(model.parameters(), lr=0.001)
jaccard_index = JaccardIndex(num_classes=2)
num_epochs = 100 

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
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
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
    if epoch % 10 == 9:
        # model.eval()
        # total_jaccard_index = 0.0
        # count = 0
        # with torch.no_grad():
        #     for images, masks in val_data_loader:
        #         images = images.to(device)
        #         masks = masks.to(device)

        #         outputs = model(images)
        #         preds = torch.sigmoid(outputs) > 0.5
        #         preds = preds.long()

        #         # JaccardIndex 계산 = segmantation 의 mIOU
        #         score = jaccard_index(preds, masks)
        #         total_jaccard_index += score
        #         count += 1
        # mean_jaccard_index = total_jaccard_index / count
        # print(f"{num_epochs} mIOU :{mean_jaccard_index} ")
        torch.save(model.state_dict(), f'{epoch+1}_model.pth')
print('Finished Train')