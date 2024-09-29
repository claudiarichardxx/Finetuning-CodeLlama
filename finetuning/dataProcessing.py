import torch
from torch.utils.data import Dataset
from dataclasses import dataclass, field
import copy
from .tokenize import Tokenize
from .utils import *

IGNORE_INDEX = -100

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

class SupervisedDataset(Dataset):
        
        """Dataset for supervised fine-tuning."""

        def preprocess(
            self, sources, targets, tokenizer, mode = 'IT'
        ):
            
            """Preprocess the data by tokenizing."""
            examples = [s + t for s, t in zip(sources, targets)]
            tt = Tokenize()
            examples_tokenized, sources_tokenized = [tt._tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
            input_ids = examples_tokenized["input_ids"]
            labels = copy.deepcopy(input_ids)

            if(mode == 'IT'):

                for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
                    label[:source_len] = IGNORE_INDEX

                return dict(input_ids=input_ids, labels=labels)
            
            else:

                return dict(input_ids=input_ids, labels=labels) 
            

        def __init__(self, data_path, tokenizer, mode = 'IT'):

            super(SupervisedDataset, self).__init__()
            print("Loading data...")
            list_data_dict = jload(data_path)

            print("Formatting inputs...")
            prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
            sources = [
                prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
                for example in list_data_dict
            ]
            targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

            print("Tokenizing inputs... This may take some time...")
            data_dict = self.preprocess(sources, targets, tokenizer, mode = mode)

            self.input_ids = data_dict["input_ids"]
            self.labels = data_dict["labels"]

        def __len__(self):
            return len(self.input_ids)

        def __getitem__(self, i):
            return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""
    tokenizer: object
    
    def __call__(self, instances):
            input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )
            labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first = True, padding_value = IGNORE_INDEX)
            return dict(
                input_ids = input_ids,
                labels = labels,
                attention_mask = input_ids.ne(self.tokenizer.pad_token_id),
            )


      
    

