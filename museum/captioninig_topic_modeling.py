import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # warning log filter

import caption_generator
import konlpy_topic_modeling

if __name__ == '__main__':
    caption_generator.main()
    konlpy_topic_modeling.main()
