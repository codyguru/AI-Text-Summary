# AI Text Summary

A Flask web application that uses the BART transformer model to automatically summarize text content.

## Features

- Summarizes long text documents  
- Handles large texts by processing in chunks  
- Simple web interface
- Real-time processing
- Cleans up temporary files and caches
- Utilizes GPU if available or CPU if it is not

## Tech Stack

- Flask
- PyTorch
- BART LLM from Hugging Face

## Project Initialization

### Requirements

To set up the project locally, you will need to create a virtual enviornment or have Docker desktop 
installed. This project utilizes Python 3.10 for the requirements.

### Venv Initialization

After forking the project, setup your virtual enviornment using venv or another desired enviornment. 
Make sure pip is updated and install the requirements.txt into the enviornment. Once successful, run 
```python app.py``` to start the application. This application runs on port 5000.

### Docker Initialization

If you prefer Docker and have Docker desktop installed, open your terminal and run ```docker pull lillidarling/ai-text-sum:latest``` to pull the image down for the container. From there you should be able to run the 
command ```docker run -d -p 5000:5000 --name summarizer lillidarling/ai-text-sum``` to start the container. 
Give it a minute to completely initialize before accessing the app in your browser. 

## Notes

1) If you run into any issues please make an issue in the repo or send me a message through discord so that I 
am aware and can correct it.

2) This project was created for fun and I have deployed it using an AWS EC2 instance. The instance 
contains the Docker image and container that spins up when the instance is activated. I connected an API 
Gateway that triggers two of three Lambda functions. The first Lambda function spins up the instance and 
will tell you if the instance is started, pending, or running, and provide health checks. The second function 
will tell you the status of the application within the instance itself. Once the application is ready, the 
function will give you the IP address to access the UI for the application. The third and final function runs 
a check to see if the instance is running. If it is, it will wait 10 minutes before stopping the instance. I
implemented this infrastructure to help mitigate costs and practice a different architecture within AWS.

3) This project has another file to calculate the disk space used of the application. I created it to figure 
out how much it was using for a deployment through PythonAnywhere. Since the BART model is large, I decided 
to deploy with AWS instead and set up the infrastructure in a way to mitigate costs as well as test an 
on-demand instance structure for another project already deployed. I implemented cleaning functions to help 
optomize the project since it didn't need to keep the cache every time it was run.

