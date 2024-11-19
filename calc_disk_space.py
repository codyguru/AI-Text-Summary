import os
import sys
import torch
from transformers import AutoTokenizer, pipeline

def get_size_format(bytes):
    """Convert bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes < 1024.0:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024.0

def calculate_model_size():
    """Calculate the disk space used by the BART model"""
    try:
        model_name = "facebook/bart-large-cnn"

        summarizer = pipeline(
            "summarization", 
            model=model_name,
            device=-1  
        )
        
        model_size = sum(p.numel() * p.element_size() for p in summarizer.model.parameters())
        
        tokenizer_size = sys.getsizeof(summarizer.tokenizer.vocab)
        
        return model_size + tokenizer_size
    except Exception as e:
        return f"Error calculating model size: {str(e)}"

def get_app_requirements_size():
    """Calculate size of requirements"""
    requirements = [
        'flask',
        'torch',
        'transformers',
        'concurrent-futures'
    ]
    
    total_size = 0
    sizes = {}
    
    for package in requirements:
        try:
            module = __import__(package)
            package_path = os.path.dirname(module.__file__)
            size = sum(
                os.path.getsize(os.path.join(dirpath, filename))
                for dirpath, _, filenames in os.walk(package_path)
                for filename in filenames
            )
            sizes[package] = size
            total_size += size
        except Exception as e:
            sizes[package] = f"Error: {str(e)}"
    
    return total_size, sizes

def get_total_disk_usage():
    """Calculate total disk usage of the application"""
    results = {
        'Model Size': calculate_model_size(),
        'App Code Size': os.path.getsize(__file__),
    }
    
    requirements_total, requirement_sizes = get_app_requirements_size()
    results['Requirements Size'] = requirements_total
    results['Individual Requirements'] = requirement_sizes
    
    results['Total Estimated Size'] = sum(
        size for size in results.values() 
        if isinstance(size, (int, float))
    )
    
    formatted_results = {}
    for key, value in results.items():
        if isinstance(value, (int, float)):
            formatted_results[key] = get_size_format(value)
        elif isinstance(value, dict):
            formatted_results[key] = {k: get_size_format(v) for k, v in value.items() if isinstance(v, (int, float))}
        else:
            formatted_results[key] = value
            
    return formatted_results

if __name__ == "__main__":
    disk_usage = get_total_disk_usage()
    
    print("\nDisk Space Analysis:")
    print("-" * 50)
    for key, value in disk_usage.items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            for sub_key, sub_value in value.items():
                print(f"  {sub_key}: {sub_value}")
        else:
            print(f"{key}: {value}")