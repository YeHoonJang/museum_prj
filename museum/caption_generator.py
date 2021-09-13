import argparse
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

import os
from datasets.tokenization_kobert import KoBertTokenizer
import json
from datasets.CustomDataset import CustomDataset
from models import caption

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # warning log filter
from datasets import coco
from configuration import Config

# dataloader 에서 batch 를 불러올 때 그 batch 데이터를 어떻게 전처리 할 지를 정의
def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    # 해당 dataloader 에서는 손상된 이미지 파일을 None 으로 처리 -> None 파일은 batch 에서 제외
    return torch.utils.data.dataloader.default_collate(batch)


@torch.no_grad()
def evaluate(model, data_loader, args, device):
    config = Config()
    model.eval()
    tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')  # kobert tokenizer 호출
    fig_idx = 0
    json_list = []

    with open(args.json_file_name + '.json', 'w', encoding="utf-8") as make_file:

        for images, caps, cap_masks, file_name in tqdm(data_loader):
            images = images.to(device)
            caps = caps.to(device)
            cap_masks = cap_masks.to(device)
            file_name = file_name
            fig = plt.figure(figsize=(20, 5))  # 이 부분 다시 확인해보기

            # predict captions
            for i in range(config.max_position_embeddings - 1):
                predictions = model(images, caps, cap_masks).to(device)
                predictions = predictions[:, i, :]
                predicted_id = torch.argmax(predictions, axis=-1)
                print(predicted_id)

                if predicted_id[0] == 102:
                    return caps

                caps[:, i + 1] = predicted_id
                cap_masks[:, i + 1] = False

            # predict 된 file, caption들을 pair단위로 list에 저장
            for i in range(len(caps)):
                result = tokenizer.decode(caps[i].tolist(), skip_special_tokens=True)
                json_list.append({"file_name": file_name[i].strip(), "caption": result.capitalize()})
                ax = fig.add_subplot(data_loader.batch_size / 2, 2, i + 1, xticks=[], yticks=[])
                imgs = images[i].cpu().numpy().transpose(1, 2, 0)
                ax.imshow(imgs)
                ax.set_title(result.capitalize())

            fig.tight_layout()

            # figure 디렉토리 유무 확인 후 생성
            if platform == "win32":
                if os.path.isdir("figure"):
                    file_path = os.path.join('figures', args.json_file_name + '_' + str(fig_idx))
                    plt.savefig(file_path)
                else:
                    os.mkdir("figure")
                    file_path = os.path.join('figures', args.json_file_name + '_' + str(fig_idx))
                    plt.savefig(file_path)
            fig_idx = fig_idx + 1

            # list에 저장된 caption을 json 파일에 작성
            json.dump({args.json_file_name: json_list}, make_file, ensure_ascii=False, indent="\t")


# if __name__ == '__main__':
def main():
    print("caption_generator")

    parser = argparse.ArgumentParser(description='Image Captioning')
    parser.add_argument('--path', type=str, default='./image', help='path to image')  # caption을 뽑고자 하는 image folder의 경로
    parser.add_argument('--v', type=str, help='version', default='v4')  # 본 모델은 torchhub에 있기 때문에 v4이상으로 지정해야 함
    parser.add_argument('--checkpoint', type=str, help='checkpoint path', default='./checkpoint3.pth')  # 한국어 COCO 데이터셋으로 훈련한 checkpoint load
    parser.add_argument('--json_file_name', type=str, help='json file name', default="image_caption")  # 저장하고자 하는 json 파일 경로
    args = parser.parse_args()

    version = args.v
    checkpoint_path = args.checkpoint
    font_path = 'C:/Windows/Fonts/gulim.ttc'
    font = font_manager.FontProperties(fname=font_path).get_name()
    rc('font', family=font)

    # if torch.cuda.is_available():
    #     device = 'cuda'
    # else:
    device = 'cpu'

    config = Config()

    # load dataloader
    predict_Dataset = CustomDataset(data_path=args.path, transform=coco.val_transform)
    sampler_val = torch.utils.data.SequentialSampler(predict_Dataset)
    predict_dataloader = DataLoader(dataset=predict_Dataset, batch_size=config.batch_size, sampler = sampler_val, collate_fn = collate_fn, drop_last=False, shuffle=False, num_workers=config.num_workers)

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
            model,_ = caption.build_model(config)
            print("Loading Checkpoint...")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model'])

    model.to(device)
    evaluate(model, predict_dataloader, args, device)