from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

class SentimentAnalyzer:
    def __init__(self, model_path='./sentiment_model'):
        print("Loading tokenizer and model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        return {"negative": probs[0][0].item(), "positive": probs[0][1].item()}
