from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import gradio as gr

class SentimentAnalyzer:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

        self.device = 0 if torch.cuda.is_available() else -1
        
        self.pipeline = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device
        )

    def analyze_sentiment(self, text):
        result = self.pipeline(text)[0]
        label = result["label"]
        score = round(result["score"], 4)
        return f"{label}: {score}"

# Load model
model_name = "/content/results/sentiment-roberta-id"  # your trained model
analyzer = SentimentAnalyzer(model_name)

# Gradio UI function
def predict(text):
    return analyzer.analyze_sentiment(text)

# Build Gradio Interface
ui = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=3, placeholder="Enter your text here..."),
    outputs="text",
    title="Sentiment Analysis",
    description=""
)

ui.launch()
  