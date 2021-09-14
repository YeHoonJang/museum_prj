# -*- coding: utf-8 -*-
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

import os
import json

from .datasets.tokenization_kobert import KoBertTokenizer
from .datasets.CustomDataset import CustomDataset
from . import caption

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # warning log filter
from .datasets import coco

# dataloader 에서 batch 를 불러올 때 그 batch 데이터를 어떻게 전처리 할 지를 정의
def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    # 해당 dataloader 에서는 손상된 이미지 파일을 None 으로 처리 -> None 파일은 batch 에서 제외
    return torch.utils.data.dataloader.default_collate(batch)


@torch.no_grad()
def evaluate(model, data_loader, args, device):
    model.eval()
    tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')  # kobert tokenizer 호출
    json_list = []

    with open(os.path.join("data", args.json_file_name) + '.json', 'w', encoding="utf-8") as make_file:
        for images, caps, cap_masks, file_name in tqdm(data_loader):
            images = images.to(device)
            caps = caps.to(device)
            cap_masks = cap_masks.to(device)

            # predict captions
            for i in range(args.max_position_embeddings - 1):
                predictions = model(images, caps, cap_masks).to(device)
                predictions = predictions[:, i, :]
                predicted_id = torch.argmax(predictions, axis=-1)

                if predicted_id[0] == 102:
                    return caps

                caps[:, i + 1] = predicted_id
                cap_masks[:, i + 1] = False

            # predict 된 file, caption들을 pair단위로 list에 저장
            for i in range(len(caps)):
                result = tokenizer.decode(caps[i].tolist(), skip_special_tokens=True)
                json_list.append({"file_name": file_name[i].strip(), "caption": result.capitalize()})

            # list에 저장된 caption을 json 파일에 작성
            json.dump({args.json_file_name: json_list}, make_file, ensure_ascii=False, indent="\t")


def main(args):
    version = args.v
    checkpoint_path = args.checkpoint

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'


    # load dataloader
    predict_Dataset = CustomDataset(data_path=args.path, transform=coco.val_transform, args=args)
    sampler_val = torch.utils.data.SequentialSampler(predict_Dataset)
    predict_dataloader = DataLoader(dataset=predict_Dataset, batch_size=args.batch_size, sampler = sampler_val, collate_fn = collate_fn, drop_last=False, shuffle=False, num_workers=args.num_workers)

    if version == 'v1':
        model = torch.hub.load('saahiluppal/catr', 'v1', pretrained=True)
    elif version == 'v2':
        model = torch.hub.load('saahiluppal/catr', 'v2', pretrained=True)
    elif version == 'v3':
        model = torch.hub.load('saahiluppal/catr', 'v3', pretrained=True)
    else:
        print("Checking for checkpoint.")
        if checkpoint_path is None:
            raise NotImplementedError('No model to chose from!')
        else:
            if not os.path.exists(checkpoint_path):
                raise NotImplementedError('Give valid checkpoint path')
            print("Found checkpoint! Loading!")
            model,_ = caption.build_model(args)
            print("Loading Checkpoint...")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model'])

    model.to(device)
    evaluate(model, predict_dataloader, args, device)