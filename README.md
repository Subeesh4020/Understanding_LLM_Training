# Understanding_LLM_Training
## From Text to Predictions: Understanding LLM Training
Large Language Models (LLMs) have become a major part of today’s AI landscape, but what actually happens behind the scenes when an LLM learns to generate text? Instead of treating an LLM as a black box, Uderstand its fundamental idea by building a very small example from scratch. In this blog, I’ll walk through the basic process step by step — from text and tokens to input-target pairs, predictions and training — to understand the core idea behind how an LLM learns to predict the next token.

## Prerequisites
Basic knowledge of Python programming is recommended to follow along with the practical examples in this blog.<br> No prior knowledge of machine learning or LLMs is required.<br>

Let’s dive into practical examples without further explanation.<br>

First import the necessary libraries<br>

```python
import random
import re
```

These libraries are used to generate random values and 
to perform any regular expression operations<br>

Now let’s use a simple sentence as training data for our LLM model<br>

```python
raw_text = '''
    A Large Language Model (LLM) 
    is an AI system that learns from 
    large amounts of text to understand 
    and generate human-like language.
'''
```

Next, let’s split the sentence into smaller pieces called tokens.<br>

```python
tokens = re.split(r'([,.:;?_!"()\']|\s)', raw_text)
```
You may notice that the list (tokens) also contains spaces and newline characters. We’ll clean these up in the next step.<br>

[‘’, ‘\n’, ‘’, ‘ ‘, ‘’, ‘ ‘, ‘’, ‘ ‘, ‘’, ‘ ‘, ‘A’, ‘ ‘, ‘Large’, …] <br>

Now remove unwanted white spaces and newlines<br>

```python
preprocessed_tokens = [word.strip() for word in tokens if word.strip()]
```
The preprocessed list will look like this: [‘A’, ‘Large’, ‘Language’, ‘Model’, …] <br>

Next Generate a vocabulary from preprocessed tokens<br>

```python
vocabulary = { token: Id for Id, token in enumerate(preprocessed_tokens) }
```
The vocabulary will look like this:<br>

{‘A’: 0, ‘Large’: 1, ‘Language’: 2, ‘Model’: 3,… }<br>

Now extract the token Ids<br>

```python
tokenIds = [vocabulary[token] for token in preprocessed_tokens]
```

Next, we will create inputs and targets from the token IDs. This is the basic idea behind self-supervised learning.<br>

Here, each input token is paired with the token that comes immediately after it. For example:<br>
Input       Target
A           Large
Large       Language
Language    Model<br>

```python
inputs = tokenIds[:-1]
targets = tokenIds[1:]
```

Now determine the vocabulary size to create an n × m matrix for storing scores during training.<br>

```python
vocab_size = len(preprocessed_tokens)
```

Generate a random n × m matrix of scores.<br>

```python
weights = [
    [random.uniform(-0.1, 0.1) for _ in range(vocab_size)]
    for _ in range(vocab_size)
]
```

The weights will look like this:<br>

                  A     Large   Language ...
       A:            [ 0.02   -0.07    0.04     .... ]
       Large:        [-0.03    0.05   -0.01     .... ]
       Language:     [ 0.06    0.01   -0.04     .... ]
       ....

Now let’s give the model the input “Large” to predict the next word and find the corresponding token Id<br>

```python
sample_text = "Large"
input_token_id = vocabulary[sample_text]
```

Now, find the score for the input token "Large" from the previously generated score matrix.<br>

```python
scores = weights[input_token_id]
```

Now consider the token Id corresponding to highest score<br>

```python
predicted_id = scores.index(max(scores))
```

At last, print the result<br>

```python
print("Predicted next word:", preprocessed_tokens[predicted_id])
print("Input word:", sample_text)
```

This is a very simplified example of the basic idea behind how an LLM learns during training.<br> In a real LLM, these calculations are performed repeatedly on a huge amount of training data.<br> The model calculates the loss, finds the gradients, and adjusts its weights to improve its predictions.<br> This process is repeated over and over until the model gradually becomes better at predicting the next token.<br>

In our simple example, because the scores are randomly generated and no actual training is performed, you may see different predicted values each time you run the script.<br> The real learning happens when we introduce the training loop, loss calculation, gradient calculation, and weight updates.<br>

Run the given script (llm_training.py) multiple times, you will see the predicted word changing.<br>


