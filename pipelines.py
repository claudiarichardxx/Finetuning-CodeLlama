from finetuning.init import *
from finetuning.finetune import *
from finetuning.dataProcessing import *
from finetuning.generate import *
from eval.humaneval import *


class Pipelines:

    def finetune(self, model_name,  mode = 'IT', output_dir = 'FT_model/', run_eval = True, run_finetuning = False, load_quantized_model = True):

        model, tokenizer = getModel(model_name, load_quantized_model = load_quantized_model)

        if(run_finetuning):

            ft = Finetune.setParameters(lora_r= 100)
            output_dir = output_dir
            ft.train(model, tokenizer, output_dir = output_dir, mode = mode, max_seq_length = 500, data_path = 'data/leetcode_instructions_code_alpaca_format.json')
            model, tokenizer = ft.merge(base_model_name = model_name, finetuned_model_dir = output_dir)
            
        
        if(run_eval):
            
            self.prompts = {'alpaca' : self.generate_alpaca_prompt, 'leetcode': self.generate_leetcode_promptV2}
            self.get_outputs(model, tokenizer, device = 'cuda', prompt_type = 'leetcode', num_samples_per_task = 1)
            pass_at_k, accuracy = HumanEval.evaluate_functional_correctness_for_n_tasks("samples.jsonl")
            return pass_at_k, accuracy
    
        return 0, 0

    def generate_alpaca_prompt(input):
            
            INSTRUCTION = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.


        ### Instruction:
        Create a Python script for this problem:
        {input}

        ### Response:"""
            
            return INSTRUCTION
    

    def generate_leetcode_prompt(input):

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
    
    def generate_leetcode_promptV2(input):

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

        '''
        reads all the problems and gets generations and saves it in "samples.jsonl"
        Input: number of samples to generate for each prompt, default = 1
        Returns: nothing

        What it does: creates a list of dictionaries with format [{task_id: generation}, {task_id: generation}]
        calls the 'generate' function to get the generated code
        writes the list of dictionaries to "samples.jsonl"

        '''
        problems = HumanEval.read_problems()
        samples = []

        if(prompt_type == None):

            for task_id in problems:
                full, completion = Generate.generate(task_id, problems[task_id]["prompt"], device, model, tokenizer) 
                samples.append(dict(task_id=task_id, full_generation = full, completion = completion))

        else:

            for task_id in problems:
                full, completion = Generate.generate(task_id, self.prompts[prompt_type](input = problems[task_id]["prompt"]), device, model, tokenizer) 

                samples.append(dict(task_id=task_id, full_generation = full, completion = completion))

        HumanEval.write_jsonl("samples.jsonl", samples)

