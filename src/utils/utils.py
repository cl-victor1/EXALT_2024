import re

def translate_to_english(proxy, tweet): # translate tweets in other languages into English 
    system_message = "As a dedicated translator specialized in tweet translation, \
                your role is to translate the provided tweet into English only if it's not already in English. \
                    Ensure that all emojis remain unchanged. If the tweet is already in English or contains only emojis, \
                        preserve the original tweet."
    role_content = []
    role_content.append(('system', system_message))
    role_content.append(('user', tweet))
    return proxy.call_chat_completion_api(role_content, model_name='gpt-4-turbo-2024-04-09')

def clean_tweet(tweet):
    tweet = tweet.replace("@user", "")
    tweet = tweet.replace("http", "")    
    tweet = replace_repeated_punctuation(tweet)
    tweet = merge_capitalized_letters(tweet)
    return tweet

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