from .dataProcessing import SupervisedDataset, DataCollatorForSupervisedDataset
from transformers import LlamaForCausalLM, CodeLlamaTokenizer, TrainingArguments
from trl import SFTTrainer,DataCollatorForCompletionOnlyLM
from peft import LoraConfig, PeftModel
import torch

class Finetune:

    def setParameters(self, lora_r= 100, lora_alpha = 16, lora_dropout = 0.1, load_quantized_model = True, output_dir = "./results", num_train_epochs = 3, batch_size = 2, weight_decay = 0.01, learning_rate = 2e-5, optim = "paged_adamw_32bit"):

        self.lora_r = lora_r  #lora rank                               
        self.lora_alpha = lora_alpha    #Alpha parameter for LoRA scaling
        self.lora_dropout = lora_dropout
        
        self.output_dir = output_dir
        self.num_train_epochs = num_train_epochs
        self.fp16 = False
        self.bf16 = False
        self.per_device_train_batch_size = batch_size
        self.per_device_eval_batch_size = batch_size
        self.gradient_accumulation_steps = 1
        self.gradient_checkpointing = True
        self.max_grad_norm = 0.3
        self.learning_rate = learning_rate #default is 2e-4
        self.weight_decay = weight_decay #0.001
        self.optim = optim
        self.lr_scheduler_type = "constant"
        self.max_steps = -1
        self.warmup_ratio = 0.03
        self.group_by_length = False
        self.logging_steps = 25
        self.max_seq_length = None
        self.packing = False
        self.save_steps = 1000000

        #Load the entire model on the GPU 0
        #device_map = {"": "cuda"} # change to 'auto' on local machine

        #Load tokenizer and model with QLoRA configuration
        self.use_4bit = load_quantized_model
        self.bnb_4bit_compute_dtype = "float16"
        self.compute_dtype = getattr(torch, self.bnb_4bit_compute_dtype)

        # Check GPU compatibility with bfloat16
        if self.compute_dtype == torch.float16 and self.use_4bit:                
                major, _ = torch.cuda.get_device_capability()
                if major >= 8:
                    print("=" * 80)
                    print("Your GPU supports bfloat16: accelerate training with bf16 = True")
                    print("=" * 80)


    def mergeModel(self, base_model_name, finetuned_model_dir):

        base_model = LlamaForCausalLM.from_pretrained(
                                                        base_model_name,
                                                        return_dict = True,
                                                        torch_dtype = torch.float16
                                                    )
        model = PeftModel.from_pretrained(base_model, finetuned_model_dir)
        model = model.merge_and_unload()
        tokenizer = CodeLlamaTokenizer.from_pretrained(finetuned_model_dir)

        return model, tokenizer


    def make_supervised_data_module(self, data_path, tokenizer, mode = 'IT'):
        """Make dataset and collator for supervised fine-tuning."""
        train_dataset = SupervisedDataset(data_path, tokenizer = tokenizer, mode = mode)
        data_collator = DataCollatorForSupervisedDataset(tokenizer = tokenizer)
        return dict(train_dataset = train_dataset, eval_dataset = None, data_collator = data_collator)  
    


    def train(self, model, tokenizer, mode = 'IT', max_seq_length = 500, output_dir ='Results/', data_path='leetcode_instructions_code_alpaca_format.json'):

        data_module = self.make_supervised_data_module(data_path, tokenizer=tokenizer, mode = mode)

        #Set supervised fine-tuning parameters
        peft_config = LoraConfig(
                                lora_alpha = self.lora_alpha,
                                lora_dropout = self.lora_dropout,
                                r = self.lora_r,
                                bias="none",
                                task_type="CAUSAL_LM",
                                )
        
        training_arguments = TrainingArguments(
              output_dir = output_dir,
              num_train_epochs = self.num_train_epochs,
              per_device_train_batch_size =self.per_device_train_batch_size,
              gradient_accumulation_steps = self.gradient_accumulation_steps,
              optim = self.optim,
              save_steps = self.save_steps,
              logging_steps = self.logging_steps,
              learning_rate = self.learning_rate,
              weight_decay = self.weight_decay,
              fp16 = self.fp16,
              bf16 = self.bf16,
              max_grad_norm = self.max_grad_norm,
              max_steps = self.max_steps,
              warmup_ratio = self.warmup_ratio,
              group_by_length = self.group_by_length,
              lr_scheduler_type = self.lr_scheduler_type
              )
        
        trainer = SFTTrainer(
            model = model,
            peft_config = peft_config,
            max_seq_length = max_seq_length,
            tokenizer = tokenizer,
            args = training_arguments,
            packing = True,
            **data_module
        )

        trainer.train()

        trainer.save_model(output_dir)
        #return trainer.model


#sentences too long
#too much content
#evolution
#data table

        
        
