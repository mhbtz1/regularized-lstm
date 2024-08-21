from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
import random
# Load the pretrained GPT-2 BPE tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

class CoQATrainDataLoader(DataLoader):

    def __init__(self, ds, tokenizer):
        self.ds = ds
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.ds['train']['story'])

    def __getitem__(self, idx):
        story = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(self.ds['train']['story'][idx]))
        questions = list(map(lambda x : tokenizer.convert_tokens_to_ids(x), self.ds['train']['questions'][idx]))
        answers = list(map(lambda x : tokenizer.convert_tokens_to_ids(x), self.ds['train']['answers'][idx]['input_text']))
        return (story, questions, answers)


    def sample_random(self):
        ranidx = random.randint(0, len(self.ds['train']['questions']))
        return (tokenizer.convert_tokens_to_ids(tokenizer.tokenize(self.ds['train']['questions'[idx]])),
                tokenizer.convert_tokens_to_ids(tokenizer.tokenize(self.ds['train']['answers'][idx]['input_text'])))

class CoQAValidateDataLoader(DataLoader):

    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds['train']['story'])
    
    def __getitem__(self, idx):
        pass