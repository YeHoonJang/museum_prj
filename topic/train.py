# -*- coding: utf-8 -*-
import pandas as pd
import gensim
from gensim import corpora

import csv
import os

from konlpy.tag import Mecab

def topic_training(args):

    captions = list()
    with open(args.stop_word) as f:
        ko_stop_words = f.read().strip()

    # 전체 data 불러오기
    with open(os.path.join(args.total_data_path, "total_datas.csv"), 'r', encoding='UTF8') as f:
        rdr = csv.reader(f)
        for line in rdr:
            if line[1] != 'text':
                captions.append(line[1])

    caption = pd.DataFrame({'caption': captions})

    # konlpy Mecab tokenizing
    mecab = Mecab()
    caption['clean_caption'] = caption['caption'].apply(lambda x: [i for i in mecab.nouns(x) if i not in ko_stop_words])

    dictionary = corpora.Dictionary(caption['clean_caption'])
    corpus = [dictionary.doc2bow(text) for text in caption['clean_caption']]

    # lda model
    lda_model = gensim.models.ldamodel.LdaModel(corpus, num_topics=args.num_topic, id2word=dictionary, passes=30)
    topics = lda_model.print_topics(num_words=args.num_word)

    # 모델 저장
    lda_model.save(os.path.join(args.total_data_path, 'model/konlpy_model_test.model'))
