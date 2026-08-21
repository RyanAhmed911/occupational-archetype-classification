import os
import re
import gradio as gr
from transformers import pipeline

MODEL_DIR = os.environ.get("MODEL_DIR", "bert_light_config2")

# Light preprocessing and max_length=150 contract from training notebook
def normalize(text):
    return re.sub(r"\s+", " ", str(text).strip())

def to_label_scores(predictions):
    return {pred["label"]: pred["score"] for pred in predictions}

def predict(text):
    normalized = normalize(text)
    if not normalized:
        return {}
    return to_label_scores(pipe(normalized, truncation=True, max_length=150)[0])

pipe = pipeline("text-classification", model=MODEL_DIR, top_k=3)

examples = [
    "A psychotherapist who uses narrative therapy to help individuals reframe their life stories.",
    "A backend engineer specializing in distributed systems and microservices architecture.",
    "A theoretical physicist researching quantum entanglement and quantum computing applications.",
    "An epidemiologist tracking disease transmission patterns and designing public health interventions."
]

interface = gr.Interface(
    predict,
    gr.Textbox(lines=4, placeholder="Describe a person's professional background and interests..."),
    gr.Label(num_top_classes=3),
    title="Persona Domain Classifier",
    description="Classify academic domain from persona descriptions",
    examples=examples
)

if __name__ == "__main__":
    interface.launch()
