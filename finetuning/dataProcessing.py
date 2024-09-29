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
            
            """
            Preprocess the data by tokenizing.

            Input arguments:
                sources (list of str): 
                    The source texts to be tokenized.
                targets (list of str): 
                    The target texts corresponding to the sources.
                tokenizer (Tokenizer): 
                    The tokenizer used for converting text to input IDs.
                mode (str, optional): 
                    The mode of operation for the preprocessing (default is 'IT' for Instruction Tuning). Options are IT or IM (Instruction Modelling).

            What it does:
                - Combines the source and target texts into a single list of examples.
                - Tokenizes the combined examples and the source texts using the provided tokenizer.
                - Creates input IDs from the tokenized examples and initializes labels based on the input IDs.
                - In 'IT' mode, replaces the parts of labels that correspond to source lengths with the `IGNORE_INDEX` token.
                - Returns a dictionary containing the input IDs and labels.

            Returns:
                dict: A dictionary containing:
                    - input_ids: The tokenized input IDs.
                    - labels: The tokenized labels (targets), modified for 'IT' or 'IM' mode.
            """
            
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

            """
            Initializes the SupervisedDataset instance by loading and processing the data.

            Input arguments:
                data_path (str): 
                    The path to the JSON file containing the dataset.
                tokenizer (Tokenizer): 
                    The tokenizer used for converting text to input IDs.
                mode (str, optional): 
                    The mode of operation for the dataset (default is 'IT' for Instruction Tuning). Options are IT or IM (Instruction Modelling).

            What it does:
                - Loads the data from the specified JSON file.
                - Formats the input prompts based on the contents of the data.
                - Constructs source texts by applying appropriate formatting from PROMPT_DICT.
                - Constructs target texts by appending the tokenizer's end-of-sequence token to the outputs.
                - Calls the preprocess method to tokenize the sources and targets.
                - Initializes input_ids and labels attributes with the processed data.

            Returns:
                None
            """

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

            """
            Returns the number of examples in the dataset.

            Input arguments:
                None

            What it does:
                - Calculates the length of the dataset by returning the number of input IDs.

            Returns:
                int: 
                    The number of examples in the dataset, which is equal to the length of the input_ids attribute.
            """
            
            return len(self.input_ids)

        def __getitem__(self, i):

            """
            Retrieves a single example from the dataset.

            Input arguments:
                i (int): 
                    The index of the example to retrieve.

            What it does:
                - Returns a dictionary containing the input IDs and labels for the specified index.

            Returns:
                dict: 
                    A dictionary with the following keys:
                    - 'input_ids': The input IDs corresponding to the specified index.
                    - 'labels': The labels corresponding to the specified index.
            """
            
            return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):

    """Collate examples for supervised fine-tuning."""
    tokenizer: object

    def __call__(self, instances):
            
            """
            Processes a batch of instances for supervised fine-tuning.

            Input arguments:
                instances (list of dict):
                    A list of dictionaries where each dictionary contains the following keys:
                    - 'input_ids': Tensor of input IDs.
                    - 'labels': Tensor of corresponding labels.

            What it does:
                - Extracts input IDs and labels from the provided instances.
                - Pads the input IDs and labels to the same length using the tokenizer's padding token for input IDs and a predefined constant for labels.
                - Constructs an attention mask to identify real tokens (not padding) in the input IDs.

            Returns:
                dict:
                    A dictionary containing:
                    - 'input_ids': A tensor of padded input IDs.
                    - 'labels': A tensor of padded labels.
                    - 'attention_mask': A tensor indicating the presence of real tokens (1) and padding (0).
            """
            
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


      
    

