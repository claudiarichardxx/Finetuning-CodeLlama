from .dataProcessing import SupervisedDataset, DataCollatorForSupervisedDataset
from transformers import LlamaForCausalLM, CodeLlamaTokenizer, TrainingArguments
from trl import SFTTrainer,DataCollatorForCompletionOnlyLM
from peft import LoraConfig, PeftModel
import torch

class Finetune:

    def setParameters(self, lora_r= 100, lora_alpha = 16, lora_dropout = 0.1, load_quantized_model = True, output_dir = "./results", num_train_epochs = 3, batch_size = 2, weight_decay = 0.01, learning_rate = 2e-5, optim = "paged_adamw_32bit"):


        """
        Set up various parameters for training and model configuration, including LoRA parameters and training settings.

        Input arguments:
            lora_r (int, optional): 
                The LoRA rank. Default is 100.
            lora_alpha (int, optional): 
                Alpha parameter for LoRA scaling. Default is 16.
            lora_dropout (float, optional): 
                Dropout rate for LoRA. Default is 0.1.
            load_quantized_model (bool, optional): 
                Whether to load the model in 4-bit quantized mode. Default is True.
            output_dir (str, optional): 
                Directory to save training outputs. Default is "./results".
            num_train_epochs (int, optional): 
                Number of training epochs. Default is 3.
            batch_size (int, optional): 
                Batch size for training and evaluation. Default is 2.
            weight_decay (float, optional): 
                Weight decay rate for optimization. Default is 0.01.
            learning_rate (float, optional): 
                Learning rate for optimization. Default is 2e-5.
            optim (str, optional): 
                The optimizer to use. Default is "paged_adamw_32bit".

        What it does:
            - Configures parameters for LoRA, including rank, alpha scaling, and dropout.
            - Sets the output directory, number of training epochs, and batch sizes.
            - Configures the learning rate, weight decay, and optimizer type.
            - Initializes the model loading configuration, including whether to load in 4-bit quantized mode.
            - Checks GPU compatibility for bfloat16 precision to optimize training performance when using 4-bit quantized models.

        Returns:
            None. This function sets class attributes based on the input arguments for use in other methods or for model configuration.
        """


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

        """
        Merge a base model with a fine-tuned model and load the corresponding tokenizer.

        Input arguments:
            base_model_name (str): 
                The name or path of the base model to be loaded.
            finetuned_model_dir (str): 
                The directory containing the fine-tuned model.

        What it does:
            - Loads the base model using the specified `base_model_name`, configured for half-precision (`float16`).
            - Loads the fine-tuned model parameters from the specified directory using `PeftModel`.
            - Merges the base model and the fine-tuned model into a single model and unloads any unnecessary parts to save memory.
            - Loads the tokenizer corresponding to the fine-tuned model from the specified directory.

        Returns:
            tuple: A tuple containing:
                - model (LlamaForCausalLM): The merged model combining the base and fine-tuned parameters.
                - tokenizer (CodeLlamaTokenizer): The tokenizer associated with the fine-tuned model.
        """

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

        """
        Create a dataset and data collator for supervised fine-tuning.

        Input arguments:
            data_path (str): 
                The path to the dataset file or directory containing the training data.
            tokenizer (Tokenizer): 
                The tokenizer to be used for tokenizing the input data.
            mode (str, optional): 
                The mode of operation for the dataset, default is 'IT' (Instruction Tuning).

        What it does:
            - Initializes a `SupervisedDataset` using the provided `data_path`, `tokenizer`, and `mode`.
            - Creates a `DataCollatorForSupervisedDataset` using the provided `tokenizer`.
            - Returns a dictionary containing the training dataset, evaluation dataset (set to None), and the data collator.

        Returns:
            dict: A dictionary with the following keys:
                - 'train_dataset': The initialized training dataset.
                - 'eval_dataset': The evaluation dataset (currently set to None).
                - 'data_collator': The data collator for batching the dataset.
        """

        train_dataset = SupervisedDataset(data_path, tokenizer = tokenizer, mode = mode)
        data_collator = DataCollatorForSupervisedDataset(tokenizer = tokenizer)
        return dict(train_dataset = train_dataset, eval_dataset = None, data_collator = data_collator)  
    


    def train(self, model, tokenizer, mode = 'IT', max_seq_length = 500, output_dir ='Results/', data_path='leetcode_instructions_code_alpaca_format.json'):

        """
        Train the model using supervised fine-tuning.

        Input arguments:
            model (PreTrainedModel): 
                The pre-trained model to be fine-tuned.
            tokenizer (Tokenizer): 
                The tokenizer used for processing the input data.
            mode (str, optional): 
                The mode of operation for the dataset (default is 'IT' for Instruction Tuning). Options are IT and IM (Instruction Modelling).
            max_seq_length (int, optional): 
                The maximum sequence length for the input data (default is 500).
            output_dir (str, optional): 
                The directory where the trained model will be saved (default is 'Results/').
            data_path (str, optional): 
                The path to the dataset file in JSON format (default is 'leetcode_instructions_code_alpaca_format.json').

        What it does:
            - Creates a data module by calling `make_supervised_data_module` with the provided `data_path` and `tokenizer`.
            - Sets up the PEFT (Parameter Efficient Fine-Tuning) configuration using `LoraConfig`.
            - Defines training arguments using `TrainingArguments`.
            - Initializes an `SFTTrainer` with the model, PEFT configuration, tokenizer, training arguments, and the data module.
            - Trains the model using the trainer.
            - Saves the trained model to the specified output directory.

        Returns:
            None
        """
        
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
        print('Training model with the configurations mentioned...')
        trainer.train()

        trainer.save_model(output_dir)
        print('Model finetuned and saved to ', output_dir)
        #return trainer.model
        
        
