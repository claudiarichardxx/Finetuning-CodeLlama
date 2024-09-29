import transformers
import copy

class Tokenize:

    def smart_tokenizer_and_embedding_resize(self,
            special_tokens_dict,
            tokenizer,
            model,
        ):
            """
            Resize the tokenizer and adjust model embeddings to include new special tokens.

            Input arguments:
                special_tokens_dict (dict): 
                    A dictionary containing the special tokens (e.g., `pad_token`, `bos_token`, `eos_token`) to add to the tokenizer.

                tokenizer (PreTrainedTokenizer): 
                    The tokenizer whose vocabulary will be resized to include new special tokens.

                model (PreTrainedModel): 
                    The model whose input and output embeddings will be resized to match the new tokenizer size.

            What it does:
                - Adds special tokens to the tokenizer using the provided `special_tokens_dict`.
                - Resizes the model's input and output token embeddings to match the tokenizer's new vocabulary size.
                - Averages the original embeddings and assigns this average to the new tokens' embeddings to avoid untrained weights.

            Returns:
                None: 
                    This function modifies the `tokenizer` and `model` in-place and does not return any value.
            
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


    def _tokenize_fn(self,strings, tokenizer):
            
            """
            Tokenize a list of strings using the provided tokenizer.

            Input arguments:
                strings (list of str): 
                    A list of input strings to be tokenized.
                
                tokenizer (PreTrainedTokenizer): 
                    A tokenizer instance (e.g., from Hugging Face) that will tokenize the input strings. 
                    This tokenizer should return PyTorch tensors (`return_tensors="pt"`).

            What it does:
                - Tokenizes each string in the input `strings` list.
                - Applies padding to ensure all tokenized outputs are of the same length.
                - Truncates tokenized strings to fit within the tokenizer's maximum length.
                - Extracts input IDs and labels from the tokenized results.
                - Computes the length of each tokenized sequence (i.e., the number of non-padding tokens).

            Returns:
                dict: A dictionary with the following keys:
                    - `input_ids` (list of torch.Tensor): The input token IDs for each string.
                    - `labels` (list of torch.Tensor): The same as `input_ids`, used as labels.
                    - `input_ids_lens` (list of int): Lengths of the input token sequences (excluding padding).
                    - `labels_lens` (list of int): Lengths of the label token sequences (excluding padding).
            """
            
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