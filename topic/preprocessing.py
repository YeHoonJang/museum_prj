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
    csv_list = glob(os.path.join(args.text_data_path, 'news/*/*.csv'))

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
    news_data_total.reset_index(inplace=True, drop=True)


    # ===================================#
    # ========TripAdvisor Reviews========#
    # ===================================#

    review_data_total = pd.DataFrame()
    review_list = glob(os.path.join(args.text_data_path, 'review/*.csv'))

    for path_ in review_list:
        # Read data
        review_dat = pd.read_csv(path_, encoding='cp949')
        review_dat.columns = ['date', 'title', 'review']

        # Text summation
        review_dat['text'] = review_dat['title'] + ' ' + review_dat['review']

        # Concatenate
        review_data_total = pd.concat([review_data_total, review_dat])
    review_data_total.reset_index(inplace=True, drop=True)


    #===================================#
    #=============Instagram=============#
    #===================================#

    insta_data_total = pd.DataFrame()
    json_list = glob(os.path.join(args.path, '*/*/*.json'))

    for path_ in json_list:
        # Data pre-processing
        insta_dat = pd.read_json(path_).transpose()

        # Concatenate
        insta_data_total = pd.concat([insta_data_total, insta_dat])

    # Content pre-processing
    insta_data_total['content'] = insta_data_total.index


    # ===================================#
    # ============Captioning=============#
    # ===================================#

    caption_data_total = pd.DataFrame()
    caption_json_list = glob(os.path.join(args.caption_data_path, '*/*.json'))

    for path_ in caption_json_list:
        with open(path_, 'r', encoding='utf-8') as f:
            pre_data = json.load(f)
            data_name = list(pre_data.keys())[0]
            caption_dat = pd.DataFrame(pre_data[data_name])

        # Data pre-processing
        caption_dat['content'] = caption_dat['file_name'].apply(lambda x: x[:x.index('.')])

        # Concatenate
        caption_data_total = pd.concat([caption_data_total, caption_dat])

    # Data pre-processing - add date series
    caption_data_total = pd.merge(caption_data_total, insta_data_total[['date', 'content']], on='content')
    caption_data_total = caption_data_total.groupby(['date', 'content'])['caption'].apply(lambda x: ', '.join(x)).reset_index()


    # ===================================#
    # ======Instagram & Caption Join=====#
    # ===================================#

    join_data_total = pd.merge(insta_data_total, caption_data_total, on=['date', 'content'], how='outer').reset_index()
    join_data_total['text'] = join_data_total.set_index('index')[['text', 'caption']].stack().groupby(level=0,sort=False).agg(', '.join)

    # ===================================#
    # ===============Total===============#
    # ===================================#

    news_data_total_text = ['date', 'text']
    review_data_total_text = ['date', 'text']
    join_data_total_text = ['date', 'text']

    total_data = pd.concat([news_data_total[news_data_total_text], review_data_total[review_data_total_text],
                            join_data_total[join_data_total_text]])


    # Save csv
    total_data.to_csv(os.path.join(args.total_data_path,"total_datas.csv"), index=False)