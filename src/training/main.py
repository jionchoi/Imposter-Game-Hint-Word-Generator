import sys
sys.path.append("..")

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
import torch
from data.load_data import PROMPT_TEMPLATE, load_hint_dataset, normalize_category

#Configuration
MODEL_NAME = "google/flan-t5-base"
OUTPUT_DIR = "../../models/hint-generator"
MAX_INPUT_LENGTH = 64
MAX_TARGET_LENGTH = 32

#Load tokenizer and model
print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

use_cuda = torch.cuda.is_available()
use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
use_fp16 = use_cuda and not use_bf16

device = "cuda" if use_cuda else "cpu"
precision = "bf16" if use_bf16 else ("fp16" if use_fp16 else "fp32")
print(f"Using device: {device} ({precision})")

# Load all datasets from datasets/ directory
dataset = load_hint_dataset("../../datasets")

print(f"Loaded {len(dataset)} examples")

# Train/test split. Since the model trains on the training set, after each train, it checks performance on the test set
# the main purpose is to see if it's just memorizing or actually learning  
dataset = dataset.train_test_split(test_size=0.15, seed=42)
print(f"Train: {len(dataset['train'])}, Test: {len(dataset['test'])}")


# Tokenization function. Tokenization converts texts input into numbers (or tokens) so that the transformer understands it

def tokenize_function(examples):
    model_inputs = tokenizer(
        examples["input"], #this will be the input. for example, "generate hint word for actions: dancing"
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding=False  # Dynamic padding handled by data collator
    )

    labels = tokenizer(
        examples["target"], #this is what the model should output. for example, "rhythem, energy, practice"
        max_length=MAX_TARGET_LENGTH,
        truncation=True, #if it takes too long, cut it off
        padding=False
    )

    #The Hugging Face Trainer expects the target output to be in a field called "labels".
    model_inputs["labels"] = labels["input_ids"]

    #       model_inputs = {                                                                                                          
    #       "input_ids": [3806, 10199, ...],      # Input: "generate hint for actions: Dancing"                                   
    #       "attention_mask": [1, 1, ...],                                                                                        
    #       "labels": [26911, 6, 1109, 6, 3944]   # ← Added here! Target: "rhythm, energy, practice"                              
    #   }

    return model_inputs

# Tokenize datasets
print("Tokenizing...")
tokenized_train = dataset["train"].map(
    tokenize_function,
    batched=True, #process multiple examples at once
    remove_columns=dataset["train"].column_names #since it only needs the tokenized inputs not the original text, we can just remove it
)

'''
Before .map()                                                                              
  {                                                                                                                         
      "input": "generate hint for actions: Dancing",                                                                        
      "target": "rhythm, energy, practice",                                                                                 
      "category": "Actions",                                                                                                
      "word": "Dancing"                                                                                                     
  }                                                                                                                         
                                                                                                                            
  After .map(                                                                                                  
  {                                                                                                                         
      "input_ids": [3806, 10199, 21, 2874, 10, 21254],                                                                      
      "attention_mask": [1, 1, 1, 1, 1, 1],                                                                                 
      "labels": [26911, 6, 1109, 6, 3944]                                                                                   
  } 
'''

tokenized_test = dataset["test"].map(
    tokenize_function,
    batched=True,
    remove_columns=dataset["test"].column_names
)

#since the example dataset has different number of tokens, so we need to pad 0 to the batches of examples so that they have consistent number of tokens
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True
)

# Training arguments. check https://huggingface.co/docs/transformers/en/main_classes/trainer
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR, #saving directory
    eval_strategy="epoch", #evaluate on test set after each epoch
    save_strategy="epoch", #save a checkpoint after each epoch
    learning_rate=2e-5, #smaller LR to reduce instability
    warmup_ratio=0.1,
    per_device_train_batch_size=8, #process 8 examples at once
    per_device_eval_batch_size=8,
    num_train_epochs=20, #better starting cap for ~4k examples; early stopping still applies
    weight_decay=0.01,
    max_grad_norm=1.0,
    save_total_limit=3, #keep a bit more history while tuning
    predict_with_generate=True, # Use actual text generation during evaluation
    fp16=use_fp16,
    bf16=use_bf16,
    logging_steps=10, #printing evey 10 steps
    load_best_model_at_end=True, #after training, load the best checkpoint (model)
    metric_for_best_model="eval_loss", #the best model is the one with lowest evaluation loss
    greater_is_better=False,
)

# Initialize trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args, #training settings
    train_dataset=tokenized_train, #data train on
    eval_dataset=tokenized_test, #data test (evaluate on)
    processing_class=tokenizer, #for decoding the output
    data_collator=data_collator, #batching and paddind
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
)

# Train
print("Starting training...")
trainer.train()

# Save final model
print(f"Saving model to {OUTPUT_DIR}/final...")
trainer.save_model(f"{OUTPUT_DIR}/final")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")

print("Done!")


# Test inference
def generate_hints(word, category="actions"):
    category = normalize_category(category)
    input_text = PROMPT_TEMPLATE.format(category=category, word=word)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_TARGET_LENGTH,
        num_beams=4,
        early_stopping=True,
        repetition_penalty=1.2,
        no_repeat_ngram_size=2
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# Quick test after training
print("\n--- Testing trained model ---")
test_words = [("Swimming", "sports"), ("Coding", "professions"), ("Cooking", "foods")]
for word, category in test_words:
    hints = generate_hints(word, category=category)
    print(f"{word} -> {hints}")
