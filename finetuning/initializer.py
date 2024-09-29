from transformers import CodeLlamaTokenizer, LlamaForCausalLM, BitsAndBytesConfig
from .tokenize import Tokenize
import torch

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"

class getModel:

    def load(self, model_name, load_quantized_model = True):

        """
        Load a model and tokenizer, with optional quantization using 4-bit precision.

        Input arguments:
            model_name (str): 
                The name or path of the pretrained model to load (e.g., a model from Hugging Face's model hub).
            
            load_quantized_model (bool, optional): 
                If True, loads the model with 4-bit quantization for memory efficiency. 
                If False, loads the model without quantization. Default is True.

        What it does:
            - Configures the model for 4-bit quantization if `load_quantized_model` is True, using parameters like quantization type (`nf4`) and compute data type (`float16`).
            - Loads the model using the specified quantization configuration if enabled.
            - Loads the tokenizer associated with the model, ensuring the correct padding side and tokenizer type.
            - If the tokenizer lacks a `pad_token`, resizes the tokenizer and model embeddings to account for it using `smart_tokenizer_and_embedding_resize`.
            - Adds special tokens (`eos_token`, `bos_token`, `unk_token`) to the tokenizer.

        Returns:
            tuple: 
                - `model` (PreTrainedModel): The loaded model, either quantized or not based on the configuration.
                - `tokenizer` (PreTrainedTokenizer): The corresponding tokenizer for the model, with special tokens added.
        """

        
        self.use_4bit = True
        self.bnb_4bit_compute_dtype = "float16"
        self.bnb_4bit_quant_type = "nf4"
        self.use_nested_quant = False
        self.compute_dtype = getattr(torch, self.bnb_4bit_compute_dtype)

        bnb_config = BitsAndBytesConfig(
                load_in_4bit = self.use_4bit,
                bnb_4bit_quant_type = self.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype = self.compute_dtype,
                bnb_4bit_use_double_quant=self.use_nested_quant,
            )
        
        if(load_quantized_model):
            model = LlamaForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=bnb_config,
                    device_map='auto')
        else:
             model = LlamaForCausalLM.from_pretrained(
                    model_name,
                    device_map='auto')

        tokenizer = CodeLlamaTokenizer.from_pretrained(
                model_name,
                padding_side = "right",
                use_fast=False,)
        
        tt = Tokenize()

        if tokenizer.pad_token is None:
                tt.smart_tokenizer_and_embedding_resize(
                    special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
                    tokenizer=tokenizer,
                    model = model,
                )

        tokenizer.add_special_tokens(
            {
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            })
        
        return model, tokenizer