from transformers import pipeline

# Initialize the BART summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(text):
    summary_text = summarizer(text, max_length=50, min_length=25, do_sample=False)[0]['summary_text']
    return summary_text

# Test the summarizer with a piece of text
test_text = """
    Your long piece of text to summarize goes here. It can be a paragraph or two from any source,
    like a news article or a scientific abstract. The goal is to see how the model compresses this text
    into a concise summary.
"""
print("Original Text:", test_text)
print("Summarized Text:", summarize_text(test_text))


