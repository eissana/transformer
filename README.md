# Transformer

In this repo we design a toy-gpt using transformers. This is based on a character-level language model. It can learn from any text given to it and generate similar texts.

I applied to Hafez poems read from a file in Farsi fonts.

1. Create an environment: `python -m venv venv`
2. Activate the environment: `source venv/bin/activate` (to deactivate just run `deactivate`)
3. Install requirements: `pip install -r requirements.txt`
4. Train a model and save it as `bin/model.pt`: `python -m train`
5. Generate text: `python -m generate`
