# -*- coding: utf-8 -*-
import gensim
from gensim import corpora
import pandas as pd
import numpy as np

import csv
import os

import warnings

from collections import defaultdict
from konlpy.tag import Mecab
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

def topic_weight_calculating(args):
    new_model = gensim.models.ldamodel.LdaModel.load(os.path.join(args.total_data_path, 'model/konlpy_model.model'))
    with open(args.stop_word) as f:
        ko_stop_words = f.read().strip()

    with open(os.path.join(args.total_data_path, "total_datas.csv"), encoding='UTF8') as f:
        rdr = csv.reader(f)
        dict_list = list()
        for line in rdr:
            if line[1] != 'text':
                dict_list.append({'date': line[0], 'caption': line[1]})

        caption = pd.DataFrame(dict_list)
        caption_concatenated = caption.groupby('date')['caption'].apply(''.join).reset_index()

        mecab = Mecab()

        caption_concatenated['clean_caption'] = caption['caption'].apply(lambda x: [i for i in mecab.nouns(x) if i not in ko_stop_words])

        dictionary = corpora.Dictionary(caption_concatenated['clean_caption'])
        corpus = [dictionary.doc2bow(text_) for text_ in caption_concatenated['clean_caption']]

        result = new_model[corpus]

        # corpus output => tuple 형태여서 이를 dataframe 형태로 풀어주는 코드
        d = defaultdict(lambda: defaultdict(int))
        for idx, rows in enumerate(result):
            for j, v in rows:
                d[j][idx] += v

        tuple_to_pandas = pd.DataFrame(d).fillna(0)
        result_df = pd.concat([caption_concatenated['date'], tuple_to_pandas], axis=1)

    result_df.to_csv(os.path.join(args.total_data_path,"konlpy_modeling_result.csv"), index=False)