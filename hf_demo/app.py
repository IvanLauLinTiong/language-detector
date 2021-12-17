from transformers import AutoModelForSequenceClassification, AutoTokenizer
from languages import LANGUANGE_MAP
import gradio as gr
import torch


model_ckpt = "ivanlau/language-detection-fine-tuned-on-xlm-roberta-base"
model = AutoModelForSequenceClassification.from_pretrained(model_ckpt)
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)


def detect_language(sentence):
    tokenized_sentence = tokenizer(sentence, return_tensors='pt')
    output = model(**tokenized_sentence)
    predictions = torch.nn.functional.softmax(output.logits, dim=-1)
    _, preds = torch.max(predictions, dim=-1)
    return LANGUANGE_MAP[preds.item()]

examples = [
    "I've been waiting for a HuggingFace course my whole life.",
    "恭喜发财!",
    "Jumpa lagi, saya pergi kerja.",
    "你食咗飯未呀?",
    "もう食べましたか?",
    "as-tu mangé",
    "أريد أن ألعب كرة الريشة"
]

inputs=gr.inputs.Textbox(placeholder="Enter your text here", label="Text content", lines=5)
outputs=gr.outputs.Label(label="Language detected:")
article = """
Fine-tuned on xlm-roberta-base model.\n
Supported languages:\n 
    'Arabic', 'Basque', 'Breton', 'Catalan', 'Chinese_China', 'Chinese_Hongkong', 'Chinese_Taiwan', 'Chuvash', 'Czech', 
    'Dhivehi', 'Dutch', 'English', 'Esperanto', 'Estonian', 'French', 'Frisian', 'Georgian', 'German', 'Greek', 'Hakha_Chin', 
    'Indonesian', 'Interlingua', 'Italian', 'Japanese', 'Kabyle', 'Kinyarwanda', 'Kyrgyz', 'Latvian', 'Maltese', 
    'Mangolian', 'Persian', 'Polish', 'Portuguese', 'Romanian', 'Romansh_Sursilvan', 'Russian', 'Sakha', 'Slovenian', 
    'Spanish', 'Swedish', 'Tamil', 'Tatar', 'Turkish', 'Ukranian', 'Welsh'
"""

gr.Interface(
    fn=detect_language,
    inputs=inputs,
    outputs=outputs,
    verbose=True,
    examples = examples,
    title="Language Detector 🔠",
    description="A simple interface to detect 45 languages.",
    article=article,
    theme="huggingface"
).launch()
