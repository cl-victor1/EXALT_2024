from model_proxy.openai_proxy import OpenAIProxy
from utils.utils import translate_to_english
from utils.utils import clean_tweet
import pandas as pd

def single_instance_process(instance):
    tweet = instance['Texts']
    tweet = translate_to_english(OpenAIProxy(), tweet)
    tweet = clean_tweet(tweet)
    return tweet

def inference():
    test_data = pd.read_csv("../data/exalt_test_participants/exalt_emotion_test_participants.tsv", sep='\t')
    test_data["Texts"] = test_data.apply(lambda x: single_instance_process(x), axis=1)
    test_data[['ID', 'Texts']].to_csv("../data/exalt_test_participants/exalt_emotion_test_processed.tsv", sep='\t', index=False)
    
if __name__ == '__main__':
    inference()