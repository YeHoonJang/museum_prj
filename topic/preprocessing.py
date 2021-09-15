import os
import json
import datetime
import pandas as pd
from glob import glob
from datetime import datetime

def text_preprocessing(args):

    #===================================#
    #============Naver News=============#
    #===================================#

    news_data_total = pd.DataFrame()
    csv_list = glob(os.path.join(args.text_data_path, '*/*.csv'))

    for path_ in csv_list:
        # Read data
        news_dat = pd.read_csv(path_)
        news_dat.columns = ['title', 'date', 'description']

        # Data processing
        news_dat['date'] = news_dat['date'].apply(lambda x: x.replace('.', '-')[:-1])

        # Text summation
        news_dat['text'] = news_dat['title'] + ' ' + news_dat['description']

        # Concatenate
        news_data_total = pd.concat([news_data_total, news_dat])

    #===================================#
    #=============Instagram=============#
    #===================================#

    insta_data_total = pd.DataFrame()
    json_list = glob(os.path.join(args.image_path, '*/*/*.json'))

    for path_ in json_list:
        # Captioning file pass (Need to remove)
        if 'caption' in path_:
            continue

        # Data pre-processing
        insta_dat = pd.read_json(path_).transpose()

        # Concatenate
        insta_data_total = pd.concat([insta_data_total, insta_dat])

    # Content pre-processing
    insta_data_total['content'] = insta_data_total.index

    #===================================#
    #============Captioning=============#
    #===================================#

    caption_data_total = pd.DataFrame()
    caption_json_list = glob(os.path.join(args.caption_data_path, '*/*.json'))

    for path_ in caption_json_list:
        with open(path_, 'r') as f:
            pre_data = json.load(f)
            data_name = list(pre_data.keys())[0]
            caption_dat = pd.DataFrame(pre_data[data_name])

        # Data pre-processing
        caption_dat['content'] = caption_dat['file_name'].apply(lambda x: x[:x.index('.')])

        # Concatenate
        caption_data_total = pd.concat([caption_data_total, caption_dat])

    # Data pre-processing
    caption_data_total = pd.merge(caption_data_total, insta_data_total[['date', 'content']], on='content')

    #===================================#
    #===============Total===============#
    #===================================#

    news_data_total_text = [['date', 'text']]
    insta_data_total_text = [['date', 'text']]