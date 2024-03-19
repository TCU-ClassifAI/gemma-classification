

import sys


def categorize_question(question: str) -> int:

    # Code inspired from https://colab.research.google.com/github/google/generative-ai-docs/blob/main/site/en/gemma/docs/pytorch_gemma.ipynb#scrollTo=GU5ZZzcZ6ik3
    # Copyright 2024 Google LLC.

    # Choose variant and machine type
    VARIANT = '2b-it' #@param ['2b', '2b-it', '7b', '7b-it', '7b-quant', '7b-it-quant']
    MACHINE_TYPE = 'cuda' #@param ['cuda', 'cpu']


    import os
    import kagglehub
    # Load model weights
    weights_dir = kagglehub.model_download(f'google/gemma/pyTorch/{VARIANT}')

    # Ensure that the tokenizer is present
    tokenizer_path = os.path.join(weights_dir, 'tokenizer.model')
    assert os.path.isfile(tokenizer_path), 'Tokenizer not found!'

    # Ensure that the checkpoint is present
    ckpt_path = os.path.join(weights_dir, f'gemma-{VARIANT}.ckpt')
    assert os.path.isfile(ckpt_path), 'PyTorch checkpoint not found!'

    # Downloading the model if it isn't already present

    
    # NOTE: The "installation" is just cloning the repo.
    # !git clone https://github.com/google/gemma_pytorch.git

    import sys

    sys.path.append('gemma_pytorch')


    from gemma_pytorch.gemma.config import get_config_for_7b, get_config_for_2b
    from gemma_pytorch.gemma.model import GemmaForCausalLM

    # ## Setup the model

    import torch

    torch.cuda.empty_cache()

    # Set up model config.
    model_config = get_config_for_2b() if "2b" in VARIANT else get_config_for_7b()
    model_config.tokenizer = tokenizer_path
    model_config.quant = 'quant' in VARIANT

    # Instantiate the model and load the weights.
    torch.set_default_dtype(model_config.get_dtype())
    device = torch.device(MACHINE_TYPE)
    model = GemmaForCausalLM(model_config)
    model.load_weights(ckpt_path)
    model = model.to(device).eval()


    # Chat templates
    USER_CHAT_TEMPLATE = '<start_of_turn>user\n{prompt}<end_of_turn>\n'
    MODEL_CHAT_TEMPLATE = '<start_of_turn>model\n{prompt}<end_of_turn>\n'

    # Load prompting data
    import pandas as pd

    data = pd.read_csv('questions.csv')
    # print(data.head())

    # Add "Question: " to the beginning of each question
    data['question'] = 'Question: ' + data['question']

    # format labels as one-character strings (0, 1, 2)
    data['label'] = data['label'].astype(str)

    # Remove the .0 from the label
    data['label'] = data['label'].str.replace('.0', '')

    # print(data.head())
    # question  label
    # 0  Question: What is the capital of California?      1
    # 1  Question: What can I do in California?      1


    def create_chat_prompt(question_df: pd.DataFrame, question) -> str:
        """
        Creates a chat prompt for a question classification task, providing
        examples and instructions for the model.

        Args:
            question_df: A DataFrame containing 'question' and 'label' columns.
            question: A string containing the question to be classified.

        Returns:
            str: A formatted chat prompt ready for model use.
        """

        CLASSIFICATION_INTRO = """You are designed to categorize questions according to Higher or Lower Order Thinking.
        Level 1 questions focus on gathering and recalling information. 
        Level 2 questions focus on making sense of gathered information, or questions focus on applying and evaluating information.
        Level 0 questions are rhetorical questions or other sorts that don't fit the model. Be careful to classify these questions correctly.
        You will be given a series of questions to classify. Please provide the classification for each question.
        You can respond with 0, 1, or 2 to indicate the classification.
        """

        USER_CHAT_TEMPLATE = '<start_of_turn>user\n{prompt}<end_of_turn>\n'
        MODEL_CHAT_TEMPLATE = '<start_of_turn>model\n{prompt}<end_of_turn>\n'

        prompt_lines = []

        # Select a random sample of questions (more efficient)
        question_df = question_df.sample(20, replace=False)
        # reindex the dataframe
        question_df = question_df.reset_index(drop=True)



        for index, row in question_df.iterrows():
            # print('row:', row, 'index:', index)

            if index == 0:  # Add intro only for the first question
                # print('row:', row)
                prompt_lines.append(USER_CHAT_TEMPLATE.format(prompt=CLASSIFICATION_INTRO + row['question']))
            else:
                prompt_lines.append(USER_CHAT_TEMPLATE.format(prompt=row['question']))

            prompt_lines.append(MODEL_CHAT_TEMPLATE.format(prompt=row['label']))

        # Add the final question for the model to classify
        prompt_lines.append(USER_CHAT_TEMPLATE.format(prompt=f'Question: {question}'))

        prompt_lines.append("<start_of_turn>model")

        return ''.join(prompt_lines)



    # Create the prompt
    prompt = create_chat_prompt(data, question)
    # print('Chat prompt:\n', prompt)

    def generate_with_retries(model, prompt, device, output_len=1, max_retries=3):
        """
        Generates a response from the model, retrying up to `max_retries` times if the response is not an integer.

        Args:
            model: The model to generate from.
            prompt: The chat prompt to use.
            device: The device to use for generation.
            output_len: The number of tokens to generate.
            max_retries: The maximum number of times to retry generation.

        Returns:
            int: The generated response.
        """
        for _ in range(max_retries):
            response = model.generate(
                USER_CHAT_TEMPLATE.format(prompt=prompt),
                device=device,
                output_len=1,
            )

            # try to format the response as an integer. If it fails, retry 3 times, then return 0
            try:
                response = int(response)
                return response
            except:
                response = 0

        return response

    # Generate a response from the model
    response = generate_with_retries(model, prompt, device)
    return response

# # Test the function
# reply = categorize_question('What is the capital of California?')
# print(reply)  # 1


def num_chars_in_question(question: str) -> int:
    return len(question)

if __name__ == "__main__":
    from flask import Flask, request, jsonify
    app = Flask(__name__)
    import os
    import json

    # print('Setting Kaggle credentials')
    # print(os.environ['KAGGLE_USERNAME'])

    # try:
    #     # export /app/secrets/kaggle.json username as KAGGLE_USERNAME and key as KAGGLE_KEY 
    #     with open('/app/secrets/kaggle.json', 'r') as file:

    #         kaggle_json = json.load(file)
    #         os.environ['KAGGLE_USERNAME'] = kaggle_json['username']
    #         print(f'KAGGLE_USERNAME: {kaggle_json["username"]}')
    #         os.environ['KAGGLE_KEY'] = kaggle_json['key']
    #     print('Kaggle credentials set')
    # except:
    #     pass

    @app.route('/categorize', methods=['POST'])
    def categorize_question_route():
        question = request.json['question']
        response = categorize_question(question)
        return jsonify({'response': response})
    
    @app.route('/healthcheck', methods=['GET'])
    def healthcheck():
        return jsonify({'status': 'ok'})
    
    app.run(port=5002, host='0.0.0.0')
    