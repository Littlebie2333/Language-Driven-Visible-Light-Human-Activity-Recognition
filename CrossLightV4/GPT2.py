import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
from torch import nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel
# from transformers import GPT2Tokenizer, GPT2Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class TextGenerator(nn.Module):
    def __init__(self, model_name='gpt2', device=None):
        super(TextGenerator, self).__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.eos_token = self.tokenizer.eos_token  # 设置 eos_token
        self.eos_token_id = self.tokenizer.eos_token_id  # 设置 eos_token
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)

    def forward(self, inputs_embeds, attention_mask=None, labels=None):
        return self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)

    def generate_ids(self, inputs_embeds, attention_mask=None, max_length=100, num_return_sequences=1, do_sample=False):
        generated_ids = self.model.generate(inputs_embeds=inputs_embeds,
                                            max_length=max_length,
                                            num_return_sequences=num_return_sequences,
                                            attention_mask=attention_mask,
                                            do_sample=do_sample)
        return generated_ids

# from transformers import AutoTokenizer, AutoModelForCausalLM
# class TextGenerator(nn.Module):
#     def __init__(self, model_name='google/gemma-2-2b-it', device=None):
#         super(TextGenerator, self).__init__()
#         self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token='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')
#         self.eos_token = self.tokenizer.eos_token  # 设置 eos_token
#         self.eos_token_id = self.tokenizer.eos_token_id  # 设置 eos_token
#         self.model = AutoModelForCausalLM.from_pretrained(
#                         "google/gemma-2-2b-it",
#                         device_map="auto",
#                         torch_dtype=torch.bfloat16,
#                         use_auth_token='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'
#                     )
#
#     def forward(self, inputs_embeds, attention_mask=None, labels=None):
#         return self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
#
#     def generate_ids(self, inputs_embeds, attention_mask=None, max_length=100, num_return_sequences=1, do_sample=False):
#         generated_ids = self.model.generate(inputs_embeds=inputs_embeds,
#                                             max_length=max_length,
#                                             num_return_sequences=num_return_sequences,
#                                             attention_mask=attention_mask,
#                                             do_sample=do_sample)
#         return generated_ids

# # Usage example:
if __name__ == "__main__":
    # Initialize the text generator
    text_gen = TextGenerator().to(device)

    text = "thrusting fist forward"
    # Tokenize the input text and convert to input embeddings
    input_ids = text_gen.tokenizer(text, return_tensors="pt").input_ids.to(text_gen.device)
    inputs_embeds = text_gen.model.transformer.wte(input_ids)  # 通过模型的嵌入层将输入 ID 转换为嵌入
    # inputs_embeds = torch.randn(1,10,768).to(device)
    print(inputs_embeds.shape)

    # # Generate text IDs
    # generated_ids = text_gen.generate_ids(inputs_embeds=inputs_embeds, max_length=100, num_return_sequences=1,
    #                                       do_sample=False)
    #
    # # Print the generated IDs
    # print(generated_ids)
    #
    # # Decode the generated text
    # generated_text = text_gen.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    # # Print the generated text
    # print(generated_text)


