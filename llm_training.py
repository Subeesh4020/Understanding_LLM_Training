# library to generate random values
import random
# Library for regular expressions
import re

# A sample sentence (we give to train the LLM model)
raw_text = '''
    A Large Language Model (LLM) 
    is an AI system that learns from 
    large amounts of text to understand 
    and generate human-like language.
'''

# Split the sentence into piece by piece called tokens
tokens = re.split(r'([,.:;?_!"()\']|\s)', raw_text)

# Remove unwanted white spaces, if you need more customization, then you can have
preprocessed_tokens = [word.strip() for word in tokens if word.strip()]
# Generate a vocabulary from token called token: Id pairs
# e.g.,
# vocabulary = {
#     "A": 0,
#     "Large": 1,
#     "Language": 2,
#     ....
# }
vocabulary = { token: Id for Id, token in enumerate(preprocessed_tokens) }

# Now store the token Ids, extract each Ids from the generated vocabulary
tokenIds = [vocabulary[token] for token in preprocessed_tokens]


# Generate some sample inputs and targets for LLM training (self supervised learning)
# e.g.,
# inputs = [0, 1, 2]
# targets = [1, 2, 3]

inputs = tokenIds[:-1]
targets = tokenIds[1:]

# Determine the vocabulary size to create an n × m matrix for storing scores
vocab_size = len(preprocessed_tokens)

# Generate random scores in an n x m matrix
weights = [
    [random.uniform(-0.1, 0.1) for _ in range(vocab_size)]
    for _ in range(vocab_size)
]

# e,g.,                 A     Large   Language ...
#     A:            [ 0.02   -0.07    0.04     .... ]
#     Large:        [-0.03    0.05   -0.01     .... ]
#     Language:     [ 0.06    0.01   -0.04     .... ]
#       ....

# Now Let's give the model an input "Large" to predict the next word
sample_text = "Large"
# Extract the Id from vocabulary
input_token_id = vocabulary[sample_text]

# Find the score for the input token "Large" from the previously generated score matrix
scores = weights[input_token_id]

# print("Scores:", scores)
# Find the token Id corresponding to highest score
predicted_id = scores.index(max(scores))

# Print the result
print("Predicted next word:", preprocessed_tokens[predicted_id])
print("Input word:", sample_text)
