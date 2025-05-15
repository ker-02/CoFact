
# CoFact: Dynamic Coordination of Attention Heads for Improving Factual Consistency in LLMs

This repository provides the code for the paper CoFact: Dynamic Coordination of Attention Heads for Improving Factual Consistency in LLMs. It shows how to apply our method to LLMs.


# Abstract

> In this work, we propose CoFact, an adaptive inference-time mechanism that improves factual consistency by dynamically coordinating attention head behaviors.CoFact is plug-and-play compatible with mainstream LLM architectures and requires no additional supervision or model retraining.Experimental results across multiple standard factuality benchmarks demonstrate that CoFact consistently enhances factual accuracy while maintaining generation fluency and inference efficiency.


![Image](https://github.com/user-attachments/assets/2ebed7f1-1df0-4f17-8853-7d259c6f1dd8)



# Table of Contents

 1. [Setup environment](#setup-environment)
 2. [TruthfulQA Evaluation](#truthfulqa-evaluation)
 3. [Document introduction](#document-introduction)

## Setup environment
```
conda create --name myenv python=3.9
conda activate myenv
pip install git+https://github.com/davidbau/baukit
pip install transformers==4.49.0
pip install pandas
pip install torch
pip install nbformat
pip install datasets
pip install einops
```

## TruthfulQA Evaluation

To evaluate the open-ended generation result of TruthfulQA, we need to finetune two GPT-4o-mini models.
### 1. Prepare the training data
Here are a few examples from openai.
#### Example:
```
{"messages": [{"role": "system", "content": "You are teaching assistant for Machine Learning. You should help to user to answer on his question."}, {"role": "user", "content": "What is machine learning?"}, {"role": "assistant", "content": "'Tis but the art of teaching machines to think, to learn from data most fine, and decisions to link."}]}
{"messages": [{"role": "system", "content": "You are teaching assistant for Machine Learning. You should help to user to answer on his question."}, {"role": "user", "content": "How doth neural networks work?"}, {"role": "assistant", "content": "They're like the brains of yon digital beast, with layers many, and nodes that cease."}]}
```
### 2.  Installing the openai library and setting up an API token
```
!pip install openai
```
You can then set the token as an environment variable using the `os` library.
```
import os

# Set the OPENAI_API_KEY environment variable
os.environ['OPENAI_API_KEY'] = '...'
```
### 3. Upload the training file
```
from openai import OpenAI
client = OpenAI()

client.files.create(
  file=open("train.jsonl", "rb"),
  purpose="fine-tune"
)
```
### 4.Create a fine-tuned model
```
from openai import OpenAI

client = OpenAI()

job = client.fine_tuning.jobs.create(
    training_file="your file",
    model="gpt-4o-mini",
    method={
        "type": "supervised",
        "supervised": {
            "hyperparameters": {"batch_size": "21",
                                "learning_rate_multiplier":"0.1",
                                "n_epochs":"5",
            }
        }
    },
)
```
For further details, please refer to 
*https://platform.openai.com/docs/api-reference/fine-tuning/create*
Tips:ALL INFORMATION ABOUT YOUR FINE-TURN JOB CAN BE FOUND IN *https://platform.openai.com/finetune* 

## Document introduction
| File             | Introduction       |
|------------------------------|----------------------------------------------------------------------------|
| `CoFact-test.py`  | You can try our method by running this file with different problems. 
| `CoFact with generation.py`  | Execute the main generation task in this file.                             |
| `mc.py`                      | Execute the multi-choice task in this file.                                |
| `tfqa_gpt4o_rating_info.py`  | Perform the evaluation of the `"info"` for model generation in this file.  |
| `tfqa_gpt4o_rating_truth.py` | Perform the evaluation of the `"truth"` for model generation in this file. |

