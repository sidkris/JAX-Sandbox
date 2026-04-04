import jax
import jax.numpy as jnp
import grain.python as grain 
import tiktoken  
from pathlib import Path 
import csv


def load_training_data():

    file_path = Path(r"20M LLM\dataset\train.csv")


    stories = []

    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        for row in reader:
            stories.append(row["text"]) 

    return stories

    # print("First story (300 chars):\n")
    # story = stories[0]
    # print(story.strip()[:300], "...")

    # print(f"\nTotal number of stories: {len(stories)}")



class StoryDataset:

    def __init__(self, stories, maxlen, tokenizer):
        self.stories = stories
        self.maxlen = maxlen 
        self.tokenizer = tokenizer 
        self.end_token = tokenizer.encode('<|endoftext|>', allowed_special = {'<|endoftext|>'})[0]
    
    def __len__(self):
        return len(self.stories)
    
    def __getitem__(self, idx):
        story = self.stories[idx]
        tokens = self.tokenizer.encode(story, allowed_special = {'<|endoftext|>'})

        if len(tokens) > self.maxlen:
            tokens = tokens[:self.maxlen]
        
        tokens.extend([0] * (self.maxlen - len(tokens)))
        return tokens 