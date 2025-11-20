from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

class SentimentAnalyzer:
    def __init__(self, model_name, tokenizer):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.pipeline = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device
        )

    def analyze_sentiment(self, text):
        result = self.pipeline(text)
        return result[0]

if __name__ == "__main__":
    model_name = "arifagustyawan/sentiment-roberta-id" # change to your model name if you have trained your own model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    sentiment_analyzer = SentimentAnalyzer(model_name, tokenizer)

    while True:
        text = input("Enter text for sentiment analysis (or 'exit' to quit): ")
        if text.lower() == 'exit':
            break

        sentiment_result = sentiment_analyzer.analyze_sentiment(text)
        print(f"Predicted Sentiment: {sentiment_result['label']} with confidence: {sentiment_result['score']}")
