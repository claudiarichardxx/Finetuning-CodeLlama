from transformers import (
    CodeLlamaTokenizer,
    LlamaForCausalLM,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
import os
import io
import json
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from trl import SFTTrainer,DataCollatorForCompletionOnlyLM
import torch
import copy
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence

from torch.utils.data import Dataset
from transformers import Trainer
import transformers

def _make_w_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode)
    return f

def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f

def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"
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

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.CodeLlamaTokenizer,
    model: transformers.LlamaForCausalLM,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.CodeLlamaTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            #max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.CodeLlamaTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.CodeLlamaTokenizer):
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
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.CodeLlamaTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(data_path, tokenizer: transformers.CodeLlamaTokenizer) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(data_path, tokenizer=tokenizer)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

#parser = transformers.HfArgumentParser((TrainingArguments))
#training_args = parser.parse_args_into_dataclasses()
lora_r = 100 # lora rank
# Alpha parameter for LoRA scaling
lora_alpha = 16
lora_dropout = 0.1
use_4bit = True
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = False
output_dir = "./results"
num_train_epochs = 10
fp16 = False
bf16 = False
per_device_train_batch_size = 4
per_device_eval_batch_size = 4
gradient_accumulation_steps = 1
gradient_checkpointing = True
max_grad_norm = 0.3
learning_rate = 2e-4 #0.001 # default is 2e-4
weight_decay = 0.01 #0.001
optim = "paged_adamw_32bit"
lr_scheduler_type = "constant"
max_steps = -1
warmup_ratio = 0.03
group_by_length = False
logging_steps = 25
max_seq_length = None
packing = False
save_steps = 1000000
    # Load the entire model on the GPU 0
    #device_map = {"": "cuda"} # change to 'auto' on local machine

    # Load tokenizer and model with QLoRA configuration
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
        load_in_4bit = use_4bit,
        bnb_4bit_quant_type = bnb_4bit_quant_type,
        bnb_4bit_compute_dtype = compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )

    # Check GPU compatibility with bfloat16
if compute_dtype == torch.float16 and use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16: accelerate training with bf16=True")
            print("=" * 80)

    # Load base model
    # AutoModelForCausalLM, AutoTokenizer
model_name = 'codellama/CodeLlama-7b-Instruct-hf'
model = transformers.LlamaForCausalLM.from_pretrained(
        model_name,
        #cache_dir=training_args.cache_dir,
        quantization_config=bnb_config,
        device_map='auto'
    )

tokenizer = transformers.CodeLlamaTokenizer.from_pretrained(
        model_name,
        #cache_dir=training_args.cache_dir,
        #model_max_length=training_args.model_max_length,
        padding_side = "right",
        use_fast=False,)

if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model = model,
        )

peft_config = LoraConfig(
      lora_alpha=lora_alpha,
      lora_dropout=lora_dropout,
      r=lora_r,
      bias="none",
      task_type="CAUSAL_LM",
      )

output_dir = 'Results/'
training_arguments = TrainingArguments(
              output_dir='Results/',
              num_train_epochs=num_train_epochs,
              per_device_train_batch_size=per_device_train_batch_size,
              gradient_accumulation_steps=gradient_accumulation_steps,
              optim=optim,
              save_steps=save_steps,
              logging_steps=logging_steps,
              learning_rate=learning_rate,
              weight_decay=weight_decay,
              fp16=fp16,
              bf16=bf16,
              max_grad_norm=max_grad_norm,
              max_steps=max_steps,
              warmup_ratio=warmup_ratio,
              group_by_length=group_by_length,
              lr_scheduler_type=lr_scheduler_type
              # report_to="tensorboard"
              )

if "llama" in model_name:
        tokenizer.add_special_tokens(
            {
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            }
        )

data_module = make_supervised_data_module(data_path='data/code_alpaca_2k.json', tokenizer=tokenizer)
    # Set supervised fine-tuning parameters
trainer = SFTTrainer(
        model= model,
        #train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=False,
        #data_collator = data_module,
        **data_module
    )
    #trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)

trainer.train()
trainer.model.push_to_hub("ClaudiaRichard/CodeLlama_Instruction_TunedV1000")
trainer.tokenizer.push_to_hub("ClaudiaRichard/CodeLlama_Instruction_TunedV1000")
