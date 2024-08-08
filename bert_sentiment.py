from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax

local_model_directory = "bert-base-multilingual-uncased-sentiment"
model = BertForSequenceClassification.from_pretrained(local_model_directory)
tokenizer = BertTokenizer.from_pretrained(local_model_directory)

def sentiment_analysis(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    probs = softmax(outputs.logits, dim=-1)
    max_prob_index = probs.argmax().item()  # Find the index of the maximum probability
    return max_prob_index
