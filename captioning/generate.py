# -*- coding: utf-8 -*-
# Import Modules
import os
import json
import tqdm
# Import PyTorch
import torch
from torch.utils.data import Dataset, DataLoader
# Import Custom Modules
from captioning.datasets.CustomDataset import CustomDataset
from captioning.datasets.tokenization_kobert import KoBertTokenizer
from captioning.datasets import coco
from captioning.utils import collate_fn
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # warning log filter

@torch.no_grad()
def evaluate(model, data_loader, args, device):
    model.eval()
    tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')
    json_list = []

    with open(args.json_file_name, 'w', encoding="utf-8") as make_file:
        for images, caps, cap_masks, file_name in tqdm.tqdm(data_loader):
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

def caption_generating(args):

    # Device setting
    device = torch.device(args.device)

    # load dataloader
    predict_Dataset = CustomDataset(data_path=args.path, transform=coco.val_transform, args=args)
    predict_dataloader = DataLoader(dataset=predict_Dataset, batch_size=args.batch_size,
                                    collate_fn=collate_fn, drop_last=False, shuffle=False, num_workers=args.num_workers)

    if args.checkpoint is None:
        raise NotImplementedError('No model to chose from!')
    else:
        if not os.path.exists(args.checkpoint):
            raise NotImplementedError('Give valid checkpoint path')

        # Model load
        model = torch.hub.load('saahiluppal/catr', args.catr_version, pretrained=False)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model'])

    model.to(device)
    evaluate(model, predict_dataloader, args, device)