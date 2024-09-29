from finetuning.initializer import *
from finetuning.finetune import *
from finetuning.dataProcessing import *
from finetuning.generate import *
from eval.humaneval import HumanEval


class Pipelines:

    def finetune(self, model_name,  mode = 'IT', output_dir = 'FT_model/', run_eval = True, 
                 run_finetuning = False, load_quantized_model = True, lora_r= 100, lora_alpha = 16, 
                 lora_dropout = 0.1, num_train_epochs = 1, batch_size = 2, weight_decay = 0.01, 
                 learning_rate = 2e-5, optim = "paged_adamw_32bit"):

        """
        Fine-tunes a language model using specified parameters and optionally evaluates the model.

        Input arguments:
            model_name (str): The name or path of the base model to be fine-tuned.
            mode (str): The mode of fine-tuning. Default is 'IT' (Instruction Tuning). Options are IT or IM (Instruction Modelling).
            output_dir (str): Directory where the fine-tuned model will be saved. Default is 'FT_model/'.
            run_eval (bool): Flag to indicate whether to run evaluation after fine-tuning. Default is True.
            run_finetuning (bool): Flag to indicate whether to perform fine-tuning. Default is False.
            load_quantized_model (bool): Flag to determine whether to load a quantized version of the model. Default is True.
            lora_r (int): Rank for LoRA. Default is 100.
            lora_alpha (int): Alpha parameter for LoRA scaling. Default is 16.
            lora_dropout (float): Dropout rate for LoRA. Default is 0.1.
            num_train_epochs (int): Number of epochs for training. Default is 1.
            batch_size (int): Batch size for training. Default is 2.
            weight_decay (float): Weight decay for optimization. Default is 0.01.
            learning_rate (float): Learning rate for optimization. Default is 2e-5.
            optim (str): Optimization algorithm to use. Default is "paged_adamw_32bit".

        What it does:
            - Loads the specified model and tokenizer.
            - If `run_finetuning` is True, sets the fine-tuning parameters and performs training on the model.
            - Merges the fine-tuned model with the base model.
            - If `run_eval` is True, evaluates the model's performance on a set of tasks and returns the results.

        Returns:
            tuple:
                - pass_at_k (float): The pass rate at K tasks, if evaluation is run.
                - accuracy (float): The accuracy of the model, if evaluation is run.
        """
        
        gg = getModel()
        model, tokenizer = gg.load(model_name = model_name, load_quantized_model = load_quantized_model)


        if(run_finetuning):

            ft = Finetune()
            ft.setParameters(lora_r = lora_r, 
                             load_quantized_model = load_quantized_model, 
                             lora_alpha = lora_alpha, 
                             lora_dropout = lora_dropout,  
                             output_dir = output_dir, 
                             num_train_epochs = num_train_epochs, 
                             batch_size = batch_size, 
                             weight_decay = weight_decay, 
                             learning_rate = learning_rate, 
                             optim = optim)
            output_dir = output_dir
            ft.train(model, tokenizer, output_dir = output_dir, mode = mode, max_seq_length = 500, data_path = 'data/leetcode_instructions_code_alpaca_format.json')
            model, tokenizer = ft.merge(base_model_name = model_name, finetuned_model_dir = output_dir)
            
        
        if(run_eval):

            he = HumanEval()

            self.prompts = {'alpaca' : self.generate_alpaca_prompt, 'leetcode': self.generate_leetcode_promptV2}
            self.get_outputs_for_first_n_tests(model, tokenizer, device = 'cuda', prompt_type = 'leetcode', num_samples_per_task = 1)
            pass_at_k, accuracy = he.evaluate_functional_correctness_for_n_tasks("samples.jsonl")
            return pass_at_k, accuracy
    
        return 0, 0

    def generate_alpaca_prompt(self, input):
            
            """
            Generates a formatted instruction prompt based on the CodeAlpaca dataset.

            Input arguments:
                input (str): The problem statement or task for which the Python script is to be created.

            What it does:
                Constructs a prompt in a specific format that includes the instruction to create a Python script
                based on the provided input problem statement. The prompt is designed to guide the model in generating
                a relevant response.

            Returns:
                str: A formatted instruction prompt for a model trained on the CodeAlpaca dataset.
            """
            
            INSTRUCTION = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.


            ### Instruction:
            Create a Python script for this problem:
            {input}

            ### Response:"""
                
            return INSTRUCTION
    

    def generate_leetcode_prompt(self, input):

        """
            Generates a formatted instruction prompt based on the leetcode dataset.

            Input arguments:
                input (str): The problem statement or task for which the Python script is to be created.

            What it does:
                Constructs a prompt in a specific format that includes the instruction to create a Python script
                based on the provided input problem statement. The prompt is designed to guide the model in generating
                a relevant response.

            Returns:
                str: A formatted instruction prompt for a model trained on the leetcode dataset.

        """

        try:
            ques = input[input.find('"""')+3:]
        except:
            ques = input[input.find("'''")+3:]

        func_header =  input[:input.find("'''")]

        INSTRUCTION = f"""Instruction: Identify the type of algorithm needed like Dynamic Programming,Sorting,Greedy,Backtracking,Divide and Conquer.


                    ### Input:
                    Write python code for this problem:
                    {ques}

                    ### Response: \n{func_header}"""
        
        return INSTRUCTION
    
    def generate_leetcode_promptV2(self, input):

        """
            Generates a formatted instruction prompt based on the leetcode dataset.

            Input arguments:
                input (str): The problem statement or task for which the Python script is to be created.

            What it does:
                Constructs a prompt in a specific format that includes the instruction to create a Python script
                based on the provided input problem statement. The prompt is designed to guide the model in generating
                a relevant response.

            Returns:
                str: A formatted instruction prompt for a model trained on the leetcode dataset.
            
        """
        
        try:
            ques = input[input.find('"""')+3:]
        except:
            ques = input[input.find("'''")+3:]

        func_header =  input[:input.find("'''")]
        INSTRUCTION = f"""Write a response that appropriately completes the request.


                    ### Input:
                    Write python code for this problem:
                    {ques}

                    ### Output: \n{func_header}"""
        
        return INSTRUCTION
    

    #to run all the problems
    def get_outputs(self, model, tokenizer, device, prompt_type = None, num_samples_per_task = 1):

        """
        Reads all the problems from the HumanEval dataset and generates outputs using the specified model and tokenizer.

        Input arguments:
            model: The language model to be used for generating outputs.
            tokenizer: The tokenizer associated with the model.
            device: The device on which the model is to be executed (e.g., 'cuda' or 'cpu').
            prompt_type (str, optional): The type of prompt to use (e.g., 'alpaca' or 'leetcode'). Defaults to None.
            num_samples_per_task (int, optional): The number of samples to generate for each task. Defaults to 1.

        What it does:
            1. Reads all problems from the HumanEval dataset.
            2. Generates outputs for each problem using the specified model and tokenizer.
            3. Supports different types of prompts based on the provided `prompt_type`.
            4. Creates a list of dictionaries in the format [{task_id: generation}, ...].
            5. Writes the generated outputs to a JSONL file named "samples.jsonl".

        Returns:
            None
        """
        
        he = HumanEval()
        problems = he.read_problems()
        samples = []
        ge = Generate()
        if(prompt_type == None):

            for task_id in problems:
                full, completion = ge.generate(task_id, problems[task_id]["prompt"], device, model, tokenizer) 
                samples.append(dict(task_id=task_id, full_generation = full, completion = completion))

        else:

            for task_id in problems:
                full, completion = ge.generate(task_id, self.prompts[prompt_type](input = problems[task_id]["prompt"]), device, model, tokenizer) 

                samples.append(dict(task_id=task_id, full_generation = full, completion = completion))

        he.write_jsonl("samples.jsonl", samples)


    #to run the first n tests
    def get_outputs_for_first_n_tests(self, model, tokenizer, device, prompt_type = None, num_of_tests = 1, num_samples_per_task = 1):

        """
        Runs the first n tests on the provided model and tokenizer and generates outputs.

        Input arguments:
            model: The language model to be used for generating outputs.
            tokenizer: The tokenizer associated with the model.
            device: The device on which the model is to be executed (e.g., 'cuda' or 'cpu').
            prompt_type (str, optional): The type of prompt to use (e.g., 'alpaca' or 'leetcode'). Defaults to None.
            num_of_tests (int, optional): The number of tests to run. Defaults to 1.
            num_samples_per_task (int, optional): The number of samples to generate for each task. Defaults to 1.

        What it does:
            1. Reads problems from the HumanEval dataset.
            2. Collects the specified number of problems for testing.
            3. Generates outputs for the selected problems using the specified model and tokenizer.
            4. Supports different types of prompts based on the provided `prompt_type`.
            5. Writes the generated outputs to a JSONL file named "samples.jsonl".

        Returns:
            None
        """
        
        he = HumanEval()
        problems = he.read_problems()
        probs = {}
        n = 0
        for key, value in problems.items():
            if(n < num_of_tests):
                probs[key] = value
            n = n + 1

        samples = []
        ge = Generate()
        if(prompt_type == None):

            for task_id in probs:
                full, completion = ge.generate(task_id, problems[task_id]["prompt"], device, model, tokenizer) 
                samples.append(dict(task_id=task_id, full_generation = full, completion = completion))

        else:

            for task_id in probs:
                full, completion = ge.generate(task_id, self.prompts[prompt_type](input = problems[task_id]["prompt"]), device, model, tokenizer) 
                samples.append(dict(task_id=task_id, full_generation = full, completion = completion))

        he.write_jsonl("samples.jsonl", samples)

