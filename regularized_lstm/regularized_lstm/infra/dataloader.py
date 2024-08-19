from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer

# Load the pretrained GPT-2 BPE tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
ds = load_dataset("stanfordnlp/coqa")

print(ds['train']['story'][0])

class CoQATrainDataLoader(DataLoader):

    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds['train']['story'])

    def __getitem__(self, idx):
        story = tokenizer.convert_tokens_to_ids(ds['train']['story'][idx])
        questions = list(map(lambda x : tokenizer.convert_tokens_to_ids(x), ds['train']['questions'][idx]))
        answers = list(map(lambda x : tokenizer.convert_tokens_to_ids(x), ds['train']['answers'][idx]['input_text']))
        pass