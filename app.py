from flask import Flask, request, render_template
from transformers import pipeline
import torch
import os
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)

# device = 0 if torch.cuda.is_available() else -1
summarizer = pipeline(
    "summarization", 
    model="facebook/bart-large-cnn",
    device=-1,
    # torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

def process_chunk(chunk):
    return summarizer(
        chunk, 
        max_length=130, 
        min_length=30, 
        do_sample=False,
        num_beams=4
    )[0]['summary_text']

@app.route('/', methods=['GET', 'POST'])
def home():
    summary = ""
    original_text = ""
    processing_time = None
    
    if request.method == 'POST':
        import time
        start_time = time.time()
        
        original_text = request.form['text']
        max_chunk = 1024
        chunks = [original_text[i:i+max_chunk] for i in range(0, len(original_text), max_chunk)]
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            summaries = list(executor.map(process_chunk, chunks))
        
        summary = ' '.join(summaries)
        processing_time = round(time.time() - start_time, 2)
    
    return render_template('index.html', 
                         summary=summary, 
                         original_text=original_text,
                         processing_time=processing_time)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)