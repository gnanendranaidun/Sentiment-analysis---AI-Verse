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

token = os.environ.get("FRIENDLI_TOKEN") or "YOUR_FRIENDLI_TOKEN"

url = "https://api.friendli.ai/dedicated/v1/chat/completions"

headers = {
  "Authorization": "Bearer " + token,
  "Content-Type": "application/json"
}

payload = {
  "model": "vxrdv29u803z",
  "messages": [
    {
      "role": "system",
      "content": "i am feeling sad, lighten up my mood\n"
    },
    {
      "role": "user",
      "content": "i am feeling sad, lighten up my mood\n"
    },
    {
      "role": "assistant",
      "content": "I'm so sorry to hear that you're feeling down! Here are some fun ways to lift your spirits:\n\n1. Watch a funny animal video - there's nothing like cute puppies or silly cats to make you smile!\n2. Listen to upbeat music - put on your favorite happy song or create a playlist of your most cheerful tunes.\n3. Call a friend - talking to someone you love can always brighten your day.\n4. Do something creative - painting, drawing, or writing can be great ways to express your emotions and feel better.\n5. Take a walk - fresh air and exercise can boost your mood and energy levels.\n6. Indulge in something you love - whether it's your favorite dessert or a favorite TV show, treat yourself to something that always makes you happy.\n\nRemember, it's okay to feel sad sometimes, but taking care of yourself and doing things that make you feel good can help."
    },
    {
      "role": "user",
      "content": ""
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