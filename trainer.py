import torch
#!pip install accelerate==0.20.1
# !pip install transformers[torch]
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, DataCollatorForLanguageModeling, Trainer, TrainingArguments, TextDataset
import re

def preprocess_text(text):
   text = text.lower() # Convert text to lowercase
   text = re.sub(r'https?://\S+|www\.\S+', '', text) # Remove URLs
   text = re.sub(r'[^A-Za-z0-9\s]+', '', text) # Remove non-alphanumeric characters
   text = re.sub(r'\s+', ' ', text).strip() # Remove extra whitespaces
   return text

file_path = "./content/chat.txt"
with open(file_path, "r", encoding="utf-8") as file:
   chat_history = file.read()

preprocessed_chat_history = preprocess_text(chat_history)

def prepare_dataset(tokenizer, chat_history, block_size=128):
  with open('chat_history.txt', 'w') as f:
      f.write(chat_history)

  dataset = TextDataset(tokenizer=tokenizer, file_path='chat_history.txt', block_size=block_size)
  return dataset

model_name = "gpt2"
config = GPT2Config.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name, config=config)

dataset = prepare_dataset(tokenizer, preprocessed_chat_history)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
   output_dir="/content/chat_history.txt",          # output directory
   num_train_epochs=3,              # total number of training epochs
   per_device_train_batch_size=4, # batch size per device during training
   per_device_eval_batch_size=4,   # batch size for evaluation
   warmup_steps=500,                # number of warmup steps for learning rate scheduler
   weight_decay=0.01,               # strength of weight decay
   logging_dir="/content/log_history.txt",            # directory for storing logs
)

trainer = Trainer(model=model, args=training_args, data_collator=data_collator, train_dataset=dataset)
trainer.train()

model.save_pretrained("./chatbot_model")
tokenizer.save_pretrained("./chatbot_model")

def generate_response(input_text):
   inputs = tokenizer.encode(input_text, return_tensors="pt")
   inputs = inputs.to('cuda')
   if inputs.shape[1] > tokenizer.model_max_length - tokenizer.num_special_tokens_to_add(pair=None):
       inputs = inputs[:, -tokenizer.model_max_length + tokenizer.num_special_tokens_to_add(pair=None):]
    
   outputs = model.generate(inputs, max_length=150, num_return_sequences=5, no_repeat_ngram_size=2, 
                             early_stopping=True, num_beams=5, temperature=0.7)

   responses = [tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True) for output in outputs]

   return responses[0]

while True:
   user_input = input("You: ")
   if user_input.lower() == "quit":
       break

   response = generate_response(user_input)
   print("Chatbot: " + response)
