import json

import pandas as pd
import tqdm
from PIL import Image

import torch
from torchvision import transforms

# TODO : 12/14 data dict 만든 다음 loader 로 부르는 거 하기!!
from vit_pytorch import ViT

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Initializing Device: {device}')

from efficientnet_pytorch import EfficientNet
model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=1) # regression 하려면 num_class=1 로 설정!

model.to(device)

n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of params: {n_parameters}")

# training setting
creterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-5)


# Preprocessing
tfms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
img = tfms(Image.open('./data/panda.jpg')).unsqueeze(0)
print(img.shape)


# Load view count
# df = pd.read_csv("./data/gj_biennale.csv")
panda_cnt = 1000

def train_one_epoch(model, criterion, optimizer, device):
    model.train()

    train_loss = 0.0
    total = 1
    with tqdm.tqdm(total=total) as pbar:
        x = img.to(device)
        y = panda_cnt.to(device)

        output = model(x)
        print(output)
output = train_one_epoch(model, creterion, optimizer, device)
# # Load ImageNet class name
# labels_map = json.load(open('./data/labels_map.txt'))
# labels_map = [labels_map[str(i)] for i in range(1000)]

# # Classify
# model.eval()
# with torch.no_grad():
#     outputs= model(img)
#     print(outputs.size())
#     Linear = torch.nn.Linear(outputs.size()[-1], 1)
#     output = Linear(outputs)
#     print(output)
#
# # Print predictions
# print("------")
# for idx in torch.topk(outputs, k=5).indices.squeeze(0).tolist():
#     prob = torch.softmax(outputs, dim=1)[0, idx].item()
#     print('{label:<75} ({p:.2f}%)'.format(label=labels_map[idx], p=prob*100))