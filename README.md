# Anchor Co-Pilot 🤖🧠

A simple mental health AI assistant. Supports creating accounts, saving simple user information to a database, including inferred emotional state. 

# Usage

## Setup

Run `pip install -r requirements.txt` to install necessary packages. 

To use OpenAI models, you will need an API key. Simply run `export OPENAI_API_KEY=<key>` in your terminal.

## Running Anchor
Run
```
python user_chat.py --model_type <'hf' or 'openai'> \
 --model_type <'/path/to/hf-model' or openai model> \
 --temperature <[0, 1.0]> \
 --max_tokens <int>
 ```

 Defaults to 
 
 `python user_chat.py --model_type 'openai' --model_type 'gpt-4o' --temperature 0.7 --max_tokens 1500`