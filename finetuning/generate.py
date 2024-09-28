import re

class Generate:


    def generate(self, taskid, prompt, device, model, tokenizer):

        '''
        This function takes the prompt as input and returns the generated text output
        Input: prompt, taskid (only for printing purposes)
        Output: the output from the Language Model

        What it does: tokenized the prompt, directly passes it to the model for generation,
        tokenizer decodes the returned tokens.
        After decoding, the precessOutput function processes the generated text and the precessed output is returned

        '''
        #model_inputs = tokenizer('#Don\'t fill more than one function' + prompt, return_tensors='pt')
        model_inputs = tokenizer(prompt, return_tensors='pt').to(device)
        print("Task ID: " + str(taskid) + "\n" + 100 * '-')
        # generate new tokens
        # greedy_output = model.generate(model_inputs['input_ids'], max_length = 1000, eos_token_id=tokenizer.eos_token_id)
        greedy_output = model.generate(model_inputs['input_ids'], max_length = 1000, eos_token_id=tokenizer.eos_token_id, pad_token_id = tokenizer.pad_token_id)
        print("Generating...")
        #max_length=500
        #, negative_prompt_attention_mask = model_inputs['attention_mask']
        
        output = tokenizer.decode(greedy_output[0])
        print('Before processing\n',output)
        
        try:
            after = self.processOutput(output)
            print('After processing\n', after)
            
        except:
            after = output
            print('Could not process')
            
        return output, after


    def processOutput(self, output):

        '''
        Function that processes the text generated by the LLM
        Input: Unprocessed text
        Output: Python code as text

        What it does:

        1. Deletes <s> tags
        2. Replaces double quotation marks (") with single quotation marks (')
        3. In case the sentence "Let's think step by step." is present, deletes the sentence
        4. HumanEval contains instructions enclosed in three single quotations - Removes the instruction lines
        5. Removes comments starting with hashtags (#)
        6. If "if __name__ == '__main__':" is presentre this phrase is taken
        7. If there are more 'def's than 'return's, then it means there is a function that started but has not ended
        (assuming that every function return, only the text befons something). So delete the extra function altogether to make the code executable
        8. If even after all this the code is not executable, there is a function that starts and there is a return statement
        but there is a statement after the function that doesn't end. there can be two cases:

        a) There are additional statements after the end of the function - delete them
        b) The return statement is not complete - delete the last function
        Returns the resultant string

        '''
            
        possible_starters = ['### Response:', '### Output:']
            
        for starter in possible_starters:
            if(starter in output):
                output = output[output.find(starter)+len(starter):]

        output = output.replace('<s>','').replace('</s>','').strip()
        output = output.replace('"',"'")
        output = output.replace('\nLet\'s think step by step.\n\n',"")
        output = output.replace('Answer:',"")
        output = self.remove_text_inside_quotes(output, "'''")
        output = self.remove_lines_starting_with_hashtag(output)
        deff = [m.start() for m in re.finditer(r"def ",output)] # finds the occurrences of 'def'(the starting indices)
        ret = [m.start() for m in re.finditer(r"return",output)] # finds the occurrences of 'return'(the starting indices)

        if("'''" in output):
            output = output[:deff[len(deff)-1]]
        if("if __name__ == '__main__':" in output):
            main = [m.start() for m in re.finditer(r"if __name__ == '__main__':",output)][0]
            output = output[:main]

        if (len(ret)==0):
            return output

        #if function starts but theres no return statement
        if (ret[len(ret)-1] < deff[len(deff)-1]):
            #print('funct not over!')
            output = output[:deff[len(deff)-1]]

        try:
            #execute the output
            exec(output)

        except:
            #Function starts and there is a return statement but there is a statement that doesn't end

            last_return = ret[len(ret)-1]
            try:
                #there are additional statements after the end of the function
                #sol : stop till the last return statement
                end = last_return + [m.start() for m in re.finditer("\n",output[last_return:])][0]
                output = output[:end]
            except:
                #the return statement is not complete
                #sol : stop till before the last def line
                output = output[:deff[len(deff)-1]]


        #print(output)
        return (output)

    
    def remove_text_inside_quotes(self, input_string, quote):

        '''
        Deletes text inside whatever is inside 'quote'
        Input: text with  comments enclosed by quotes
        Output: text without the comments within quotes

        What it does: Uses regular expression to delete the required text

        '''

        exp = quote +'(.*?)' + quote
        pattern = re.compile(exp, re.DOTALL)
        result = re.sub(pattern, '', input_string)
        return result
    
    def remove_lines_starting_with_hashtag(self, input_string):

        '''
        Deletes lines starting with hashtag

        Input: text with  lines that start with hashtag
        Output: text without lines that start with hashtag


        What it does: reads string line by line and deletes lines that start with hashtag

        '''

        lines = input_string.split('\n')
        filtered_lines = [line for line in lines if not line.startswith('#')]
        result_string = '\n'.join(filtered_lines)
        return result_string