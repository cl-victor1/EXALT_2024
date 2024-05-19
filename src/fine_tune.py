'''
This script fine-tunes gpt-3.5-turbo on the training data, and prints the id of this fine-tuned model in the command line.
'''
from openai import OpenAI
import json
import argparse
import re

def replace_repeated_punctuation(text): # NB: this function might have adverse impact 
    # Define a regular expression pattern to match repeated punctuation separated by spaces
    pattern = r'(\s*[\.,;:!?]+\s*)+'
    # Replace occurrences of the pattern with a single punctuation mark
    replaced_text = re.sub(pattern, lambda m: m.group(1)[0] + " ", text)
    return replaced_text

def merge_capitalized_letters(input_string):
    def replace_match(match):
        # Get the matched substring
        matched_string = match.group(0)        
        # Compress uppercase letters separated by spaces
        replacement_string = re.sub(" ", "", matched_string)
        return replacement_string
    # Define the pattern
    pattern = r'\b[A-Z](?:\s[A-Z])*\b'  # Pattern to match a string with uppercase letters separated by spaces
    return re.sub(pattern, replace_match, input_string)

def clean_tweet(tweet):
    tweet = tweet.replace("@user", "")
    tweet = tweet.replace("http", "")    
    tweet = replace_repeated_punctuation(tweet)
    tweet = merge_capitalized_letters(tweet)
    return tweet
    
def preprocess(input_file, training_file, validation_file): # turn the training file (.tsv) into conversational chat format (.jsonl)
    jsonl_data = []
    with open(input_file, "r") as input:
        line_num = 0
        for line in input.readlines():
            if line_num > 0: # skip the header
                items = line.strip().split('\t')
                tweet, label = items[1], items[2]
                tweet = clean_tweet(tweet) # clean tweets
                json_line = {
                    "messages": [
                        {"role": "system", "content": "As a supportive assistant specialized in tweet classification,\
                            you're tasked with determining the emotion conveyed in a given tweet. Utilizing your intuitive understanding, \
                                analyze the sentiment of the provided tweet. Your response should be just one word,\
                                    choosing one emotion from these 6 emotions: Love, Joy, Anger, Fear, Sadness, Neutral."},
                        {"role": "user", "content": tweet},
                        {"role": "assistant", "content": label}
                    ]
                }
                jsonl_data.append(json_line)
            line_num += 1
    # training_data : validation_data = 4 : 1 (only in terms of fine-tuning)
    with open(training_file, "w") as training, open(validation_file, "w") as validation:
        line_num = 1
        for line in jsonl_data:
            if line_num <= 4000: # output training data            
                training.write(json.dumps(line) + "\n")
            else: # output validation data
                validation.write(json.dumps(line) + "\n")
            line_num += 1
            
def tune_model(training_file_name, validation_file_name):
    # Upload training_data and validation_data
    client = OpenAI()
    training_file_id = client.files.create(
    file=open(training_file_name, "rb"),
    purpose="fine-tune"
    ).id  
    validation_file_id = client.files.create(
    file=open(validation_file_name, "rb"),
    purpose="fine-tune"
    ).id  
    # Create a fine-tuned model
    tuned_model_id = client.fine_tuning.jobs.create(
    training_file=training_file_id, 
    validation_file=validation_file_id, 
    model="gpt-3.5-turbo",
    hyperparameters={
        # could try more hyperparameters combinations
        "learning_rate_multiplier": 2,
        "batch_size": 8,
        "n_epochs": 3        
        },
    suffix="first-tuning"
    ).id
    return tuned_model_id    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="../../data/exalt_emotion_train.tsv")
    parser.add_argument("--training_file", type=str, default="data/train.jsonl")
    parser.add_argument("--validation_file", type=str, default="data/dev.jsonl")
    args = parser.parse_args()
    
    # generate training data and validation data
    preprocess(args.input_file, args.training_file, args.validation_file) 
    
    # the following codes can only run on Colab since connection errors persist on the local machine
    tuned_model_id = tune_model(args.training_file, args.validation_file) 
    print(tuned_model_id)
            
if __name__ == '__main__':
    main()