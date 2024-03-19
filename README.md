# Sample build command
`docker build -t gemma_pytorch:latest .`
`docker run -p 5002:5002 --gpus all -v ~/.kaggle/kaggle.json:/root/.kaggle/kaggle.json gemma_pytorch:latest`

# Sample request
`curl -X POST -H "Content-Type: application/json" -d '{"question":"What is the square root of 5?"}' http://localhost:5002/categorize`