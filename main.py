from flask import Flask
import torch
from flask import request, jsonify
from transformers import T5TokenizerFast 
from summarization.transfer import SummaryModule, Summary
import re 
import pandas as pd

app = Flask(__name__)

model_save_name = 'summary_t5transformer_5_epochs_pretrained.pt'
model_pretrained = 't5-base'

Model = SummaryModule(model_pretrained)
Tokenizer = T5TokenizerFast.from_pretrained(model_pretrained)
Summarizer = Summary(Model, Tokenizer)

@app.route("/hello")
def hello():
    return "Hello World!"

@app.route('/', methods=['GET'])
def make_summary():
    return jsonify()



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)