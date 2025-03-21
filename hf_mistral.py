# import transformers
# import torch

# model_name_or_path = "m42-health/Llama3-Med42-8B"

# pipeline = transformers.pipeline(
#     "text-generation",
#     model=model_name_or_path,
#     torch_dtype=torch.bfloat16,
#     device_map="auto",
# )

# messages = [
#     {
#         "role": "system",
#         "content": (
#             "You are a helpful, respectful and honest medical assistant. You are a second version of Med42 developed by the AI team at M42, UAE. "
#             "Specify the Emotion, Tone, of the user. "
#             "Always answer as helpfully as possible, while being safe. "
#             "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. "
#             "Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. "
#             "If you don’t know the answer to a question, please don’t share false information."
#         ),
#     },
#     {"role": "user", "content": "What are the symptoms of diabetes?"},
# ]

# prompt = pipeline.tokenizer.apply_chat_template(
#     messages, tokenize=False, add_generation_prompt=False
# )

# stop_tokens = [
#     pipeline.tokenizer.eos_token_id,
#     pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
# ]

# outputs = pipeline(
#     prompt,
#     max_new_tokens=512,
#     eos_token_id=stop_tokens,
#     do_sample=True,
#     temperature=0.4,
#     top_k=150,
#     top_p=0.75,
# )

# print(outputs[0]["generated_text"][len(prompt) :])
import requests
import os

token = os.environ.get("FRIENDLI_TOKEN") or "flp_J8ORWjDgjjLRp5pPWlAPODZlFr5AIQCjLvh9wuZPAS99d"

url = "https://api.friendli.ai/dedicated/v1/chat/completions"

headers = {
  "Authorization": "Bearer " + token,
  "Content-Type": "application/json"
}

payload = {
  "model": "vxrdv29u803z",
  "messages": [
    {
      "role": "user",
      "content": "i am feeling sad, lighten up my mood\n"
    }
  ],
  "max_tokens": 2048,
  "top_p": 0.8,
  "stream": True,
  "stream_options": {
    "include_usage": True
  }
}

response = requests.request("POST", url, json=payload, headers=headers)

print(response.text)