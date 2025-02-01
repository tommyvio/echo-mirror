import os
import uuid
import whisper
import torch
from flask import Flask, render_template, request, jsonify
from transformers import MarianMTModel, MarianTokenizer
import pyttsx3
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = whisper.load_model("small").to(device)

def get_translation_model(lang_code):
    model_name = f"Helsinki-NLP/opus-mt-en-{lang_code}"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return model, tokenizer

engine = pyttsx3.init()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file uploaded'}), 400

    audio_file = request.files['audio']
    target_lang = request.form.get('language', 'es') 

    filename = secure_filename(audio_file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    audio_file.save(file_path)

    try:
        audio_transcription = whisper_model.transcribe(file_path)["text"]
    except Exception as e:
        return jsonify({'error': f'Speech-to-text failed: {str(e)}'}), 500

    try:
        translation_model, translation_tokenizer = get_translation_model(target_lang)
        input_tokens = translation_tokenizer(audio_transcription, return_tensors="pt", padding=True)
        translation_ids = translation_model.generate(**input_tokens)
        translated_text = translation_tokenizer.decode(translation_ids[0], skip_special_tokens=True)
    except Exception as e:
        return jsonify({'error': f'Translation failed: {str(e)}'}), 500

    response_audio = f"static/{uuid.uuid4()}.mp3"
    engine.save_to_file(translated_text, response_audio)
    engine.runAndWait()

    return jsonify({'original_text': audio_transcription, 'translated_text': translated_text, 'audio': response_audio})

if __name__ == '__main__':
    app.run(debug=True)
