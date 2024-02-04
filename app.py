from flask import Flask, jsonify, request, render_template
from transformers import pipeline
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from collections import Counter
import string
import PyPDF2
import docx


app = Flask(__name__)

# Load the BART summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/summarize', methods=['POST'])
def summarize_text():
    data = request.json
    text = data.get('text')
    style = data.get('style', 'narrative')  # Assuming style is passed in the request

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    try:
        summary = summarizer(text, max_length=100, min_length=30, do_sample=False)[0]['summary_text']

        if style == 'bullet_points':
            summary = transform_to_bullet_points(summary)

        return jsonify({'original': text, 'summary': summary})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def add_analytical_tone(text):
    # Basic keyword triggers for analytical comments
    cause_keywords = ['increase', 'decrease', 'rise', 'fall', 'growth']
    effect_keywords = ['impact', 'effect', 'result', 'consequence']

    # Split the text into sentences
    sentences = text.split('. ')

    # Analyze each sentence and append an analytical comment if relevant
    analytical_sentences = []
    for sentence in sentences:
        if any(keyword in sentence for keyword in cause_keywords):
            analytical_comment = " This suggests underlying factors that may be influencing these changes."
            analytical_sentences.append(sentence + analytical_comment)
        elif any(keyword in sentence for keyword in effect_keywords):
            analytical_comment = " This highlights the potential outcomes and their significance."
            analytical_sentences.append(sentence + analytical_comment)
        else:
            analytical_sentences.append(sentence)

    # Combine sentences back into a paragraph
    analytical_summary = '. '.join(analytical_sentences)

    return analytical_summary


def transform_to_bullet_points(text):
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)

    # Extracting keywords to identify key sentences
    words = [word.lower() for sentence in sentences for word in sentence.split() if word.isalpha()]
    words = [word for word in words if word not in stopwords.words('english')]
    word_frequencies = Counter(words)
    most_common_words = [word for word, freq in word_frequencies.most_common(5)]

    # Select sentences that contain the most common keywords
    key_sentences = [sentence for sentence in sentences if
                     any(common_word in sentence.lower() for common_word in most_common_words)]

    # Format sentences as bullet points
    bullet_points = "\n".join(["- " + sentence for sentence in key_sentences])

    return bullet_points

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        # Process the uploaded file
        text = extract_text_from_file(file)
        summary = summarizer(text, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
        return jsonify({'summary': summary})

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'pdf', 'docx'}

def extract_text_from_file(file):
    # Extract text from the uploaded file
    if file.filename.endswith('.pdf'):
        reader = PyPDF2.PdfFileReader(file)
        text = ' '.join([reader.getPage(i).extractText() for i in range(reader.numPages)])
    elif file.filename.endswith('.docx'):
        doc = docx.Document(file)
        text = ' '.join([paragraph.text for paragraph in doc.paragraphs])
    else:
        text = ''
    return text



if __name__ == '__main__':
    app.run(debug=True)
