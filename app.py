from flask import Flask, request, render_template, jsonify
from transformers import pipeline
import torch
import os
from concurrent.futures import ThreadPoolExecutor
import gc
import logging
import psutil
import sys
import shutil
import tempfile

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

model_ready = False
summarizer = None

def log_system_resources():
    process = psutil.Process(os.getpid())

    memory_usage = process.memory_info().rss / 1024 / 1024
    logger.info(f"Memory usage: {memory_usage:.2f} MB")
    
    disk = psutil.disk_usage('/')
    logger.info(f"Disk usage - Total: {disk.total/1024/1024/1024:.1f}GB, "
               f"Used: {disk.used/1024/1024/1024:.1f}GB, "
               f"Free: {disk.free/1024/1024/1024:.1f}GB")

def cleanup_resources():
    """Clean up system resources including temporary files and caches"""
    try:
        cache_dir = os.path.expanduser("~/.cache/huggingface")
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir, ignore_errors=True)
            logger.info("Cleared Hugging Face cache")
        
        temp_dir = tempfile.gettempdir()
        for filename in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                logger.error(f'Failed to delete {file_path}. Reason: {e}')
        
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logger.info("Resource cleanup completed")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")


def initialize_model():
    global model_ready, summarizer
    logger.info("Initializing model...")

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            device = 0
            logger.info("GPU detected - using CUDA")
            dtype = torch.float16 
        else:
            device = -1 
            logger.info("No GPU detected - using CPU")
            dtype = torch.float32 
        
        cleanup_resources()
        log_system_resources()
        
        summarizer = pipeline(
            "summarization", 
            model="facebook/bart-large-cnn",
            device=device,
            model_kwargs={"torch_dtype": dtype}
        )
        model_ready = True
        logger.info(f"Model initialized successfully on {'GPU' if device == 0 else 'CPU'}")
        log_system_resources()
    except Exception as e:
        logger.error(f"Error initializing model: {e}")
        model_ready = False


from threading import Thread
init_thread = Thread(target=initialize_model)
init_thread.start()


def process_chunk(chunk):
    try:
        if len(chunk.strip()) == 0:
            return ""
            
        max_length = min(130, max(30, len(chunk) // 3))
        min_length = min(30, max(10, len(chunk) // 4))
        
        return summarizer(
            chunk, 
            max_length=max_length,
            min_length=min_length, 
            do_sample=False,
            batch_size=1,
            num_beams=2
        )[0]['summary_text']
    except Exception as e:
        logger.error(f"Error processing chunk: {str(e)}")
        return ""
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_ready': model_ready
    })

@app.route('/')
def home():
    if not model_ready:
        return render_template('loading.html')
    return render_template('index.html')

@app.route('/status')
def status():
    if model_ready:
        return jsonify({
            'status': 'ready',
            'state': 'ready',
            'message': 'Application is ready'
        })
    return jsonify({
        'status': 'initializing',
        'state': 'initializing',
        'message': 'Model is still initializing. Please wait...'
    })

@app.route('/summarize', methods=['GET', 'POST'])
def summarize():
    if not model_ready:
        return jsonify({
            'error': 'Model is not ready yet. Please wait.'
        }), 503
    
    summary = ""
    original_text = ""
    processing_time = None
    error_message = None
    
    if request.method == 'POST':
        try:
            import time
            start_time = time.time()
            log_system_resources()
            
            original_text = request.form['text']
            if not original_text:
                return render_template('index.html', 
                                     error_message="Please enter some text to summarize.")
            
            max_chunk = 1024 if torch.cuda.is_available() else 512
            chunks = [original_text[i:i+max_chunk] for i in range(0, len(original_text), max_chunk)]
            chunks = [c for c in chunks if c.strip()]
            
            logger.info(f"Processing {len(chunks)} chunks")
            
            max_workers = 3 if torch.cuda.is_available() else 2
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                summaries = list(executor.map(process_chunk, chunks))
            
            summary = ' '.join(filter(None, summaries))
            processing_time = round(time.time() - start_time, 2)
            
            cleanup_resources()
            log_system_resources()
            
        except Exception as e:
            logger.error(f"Error in processing: {str(e)}")
            error_message = "An error occurred while processing your request. Please try again."
    
    return render_template('index.html', 
                         summary=summary, 
                         original_text=original_text,
                         processing_time=processing_time,
                         error_message=error_message)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)