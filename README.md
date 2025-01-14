# Finetuning-CodeLlama

# Installation
To  clone this repository:
```
git clone https://github.com/claudiarichardxx/Finetuning-CodeLlama.git
```

# To setup the environment:
To  install the required libraries, run this:
```
pip install -r requirements.txt
```
# Quick start
To setup your environment and run, you can find step by step code implementations in /finetuning-codellama.ipynb

# Description
The aim of this study is to explore Parameter Efficient Fine-Tuning using the leetcode dataset, data preparation techniques(Prompt templates, data generation, etc), and evaluation methods. Additionally, given the proven improvements in performance with Instruction Modelling (IM), experiments are conducted with both IM and Instruction Tuning. The dataset used is the Leetcode dataset, with the possibility for users to add other datasets in the CodeAlpaca format. All fine-tuning results presented in this work is performed using qLoRA, though options are available for LoRA and Full Fine-Tuning (FFT) as well. The code for Evol-Instruct is sourced partially from the official repository [35]. Provisions are made for evaluation using HumanEval and MBPP datasets.

This can be summarised as follows:
• Training Dataset: Leetcode
• PeFT method: qLoRA
• Data Generation: Evol-Instruct
• Objectives: Instruction Tuning, Instruction Modelling (Loss over Instructions)
• Evaluation Datasets: HumanEval, MBPP

# Objectives

## Instruction Tuning: The Instruction Tuning objective is to focus only on generating the output code. The instruction and input should not be included in the target; instead, they should be masked using the ‘ignore index’ token (set to -100 for CodeLlama). The source consists of the instruction and input, while the target is the output.

## Instruction Modelling: The Instruction Modelling objective, on the other hand, aims to generate both the instruction and the code, so no masking is required. Here, the source includes the instruction and input, and the target includes the instruction, input, and output together.

# The configuration of our run was
```
num_train_epochs = 3
fp16 = False
bf16 = False
per_device_train_batch_size = 2
per_device_eval_batch_size = 2
gradient_accumulation_steps = 1
gradient_checkpointing = True
max_grad_norm = 0.3
learning_rate = 2e-5   # default is 2e-4
weight_decay = 0.01   # 0.001
optim = 'paged_adamw_8bit'
lr_scheduler_type = "constant"
max_steps = -1
warmup_ratio = 0.03
group_by_length = False
logging_steps = 25
max_seq_length = None
packing = False
```

