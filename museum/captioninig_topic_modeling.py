import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # warning log filter

import caption_generator
import konlpy_topic_modeling

if __name__ == '__main__':
    print("main")
    caption_generator.main()
    print("caption done")
    konlpy_topic_modeling.main()
    print("tm done")
