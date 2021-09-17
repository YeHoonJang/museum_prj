import argparse
import warnings
warnings.filterwarnings(action='ignore')
from time import time

from captioning.train import captioning_training
from captioning.generate import caption_generating
from topic.train import topic_training
from topic.generate import topic_weight_calculating

def main(args):

    # Path setting
    path_check(args)

    # Time setting
    total_start_time = time()
    # Task run
    if args.captioning_training:
        captioning_training(args)

    if args.caption_generating:
        caption_generating(args)

    if args.topic_training:
        topic_training(args)
    
    if args.topic_weight_calculate:
        topic_weight_calculating(args)

    if args.regression_ensemble:
        regression_ensemble(args)

    # Time calculate
    print(f'Done! ; {round((time()-total_start_time)/60, 3)}min spend')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parsing Method")

    # Task setting
    parser.add_argument('--captioning_training', action='store_true')
    parser.add_argument('--caption_generating', action='store_true')
    parser.add_argument('--topic_training', action='store_true')
    parser.add_argument('--topic_weight_calculating', action='store_true')
    parser.add_argument('--resume', action='store_true')

    # Image Captioning
    parser.add_argument('--path', type=str, default='/HDD/dataset/sba_museum/image',
                        help='path to image')
    parser.add_argument('--catr_version', type=str, default='v3', choices=['v1', 'v2', 'v3', 'v4'],
                        help='version')  # 본 모델은 torchhub에 있기 때문에 v4이상으로 지정해야 함

    parser.add_argument('--json_file_name', type=str, help='json file name', default="/HDD/dataset/sba_museum/image_caption.json")  # 저장하고자 하는 json 파일 경로

    # Learning Rates
    parser.add_argument('--lr_backbone', type=float, help='backbone learning rate', default=1e-5)
    parser.add_argument('--lr', type=float, help='learning rate', default=1e-4)

    # Epochs
    parser.add_argument('--epochs', type=int, help='epoch', default=30)
    parser.add_argument('--lr_drop', type=int, help='learning rate drop', default=20)
    parser.add_argument('--weight_decay', type=float, help='weight decay', default=1e-4)

    # Backbone
    parser.add_argument('--backbone', type=str, help='backbone', default='resnet101')
    parser.add_argument('--position_embedding', type=str, help='position embedding', default='sine')
    parser.add_argument('--dilation', type=bool, help='dilation', default=True)

    # Basic
    parser.add_argument('--device', type=str, help='device type', default='cpu')
    parser.add_argument('--seed', type=int, help='seed', default=42)
    parser.add_argument('--batch_size', type=int, help='batch size', default=32)
    parser.add_argument('--num_workers', type=int, help='number of workers', default=4)
    parser.add_argument('--checkpoint', type=str, help='checkpoint path', default='/HDD/dataset/sba_museum/checkpoint3.pth')  # 한국어 COCO 데이터셋으로 훈련한 checkpoint load
    parser.add_argument('--clip_max_norm', type=float, help='clip max norm', default=0.1)

    # Transformer
    parser.add_argument('--hidden_dim', type=int, help='hidden dim', default=256)
    parser.add_argument('--pad_token_id', type=int, help='pad token id', default=0)
    parser.add_argument('--max_position_embeddings', type=int, help='max position embeddings', default=128)
    parser.add_argument('--layer_norm_eps', type=float, help='layer normalization epsilon', default=1e-12)
    parser.add_argument('--dropout', type=float, help='dropout', default=0.1)
    parser.add_argument('--vocab_size', type=int, help='vocabulary size', default=30522)

    parser.add_argument('--enc_layers', type=int, help='encoding layers', default=6)
    parser.add_argument('--dec_layers', type=int, help='decoding layers', default=6)
    parser.add_argument('--dim_feedforward', type=int, help='feedforward dimension', default=2048)
    parser.add_argument('--nheads', type=int, help='number of heads', default=8)
    parser.add_argument('--pre_norm', type=bool, help='pre norm', default=True)

    # Dataset
    parser.add_argument('--image_dir', type=str, help='coco image path', default='/HDD/dataset/coco_2014')
    parser.add_argument('--anno_dir', type=str, help='annotation path', default='/HDD/dataset/caption_ko')
    parser.add_argument('--text_data_path', type=str, help='text data path', default='/HDD/dataset/sba_museum/text')
    parser.add_argument('--caption_data_path', type=str, help='caption data path', default='/HDD/dataset/sba_museum/caption')
    parser.add_argument('--total_data_path', type=str, help='total preprocessed data path',default='/HDD/dataset/sba_museum/')
    parser.add_argument('--limit', type=int, help='limit', default=-1)

    args = parser.parse_args()
    main(args)
