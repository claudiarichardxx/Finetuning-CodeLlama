import transformers
import copy

class Tokenize:

    def smart_tokenizer_and_embedding_resize(
            special_tokens_dict,
            tokenizer,
            model,
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


    def _tokenize_fn(strings, tokenizer):
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