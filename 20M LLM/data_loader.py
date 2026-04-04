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
    


def print_sampler_example(sampler, name):
    print(f"\n{name}")
    for i, idx in enumerate(sampler):
        print(f"Record {i} : {idx}")


def create_dataloader(stories, tokenizer, maxlen, batch_size, shuffle = False, num_epochs = 1, seed = 21, worker_count = 0):
    dataset = StoryDataset(stories, maxlen, tokenizer)
    estimated_batches = len(dataset) // batch_size

    sampler = grain.IndexSampler(
        num_records = len(dataset),
        shuffle = shuffle,
        seed = seed,
        shard_options = grain.NoSharding(),
        num_epochs = num_epochs
    )

    dataloader = grain.DataLoader(
        data_source = dataset,
        sampler = sampler,
        operations = [grain.Batch(batch_size = batch_size, drop_remainder = True)], 
        worker_count = worker_count
    )

    return dataloader, estimated_batches

