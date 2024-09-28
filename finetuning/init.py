from transformers import CodeLlamaTokenizer, LlamaForCausalLM, BitsAndBytesConfig
from .tokenize import smart_tokenizer_and_embedding_resize, _tokenize_fn

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"

class getModel:

    def __init__(self, model_name, load_quantized_model = True):

        self.use_4bit = True
        self.bnb_4bit_compute_dtype = "float16"
        self.bnb_4bit_quant_type = "nf4"
        self.use_nested_quant = False

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

        if tokenizer.pad_token is None:
                smart_tokenizer_and_embedding_resize(
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