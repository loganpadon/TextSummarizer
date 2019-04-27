import pandas as pd
import re
from nltk.corpus import stopwords
from pickle import dump, load
from retrieve_article import get_articles
import tensorflow as tf
import math
import pickle
from keras import Input, Model
from keras.layers import Dense
from keras.layers import LSTM

# articles = get_articles()
#
# vocabulary_size, embedding_size, batch_size, num_sampled = 0
#
# embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
# nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],stddev=1.0 / math.sqrt(embedding_size)))
# nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
#
# # Placeholders for inputs
# train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
# train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
#
# embed = tf.nn.embedding_lookup(embeddings, train_inputs)
#
# # Compute the NCE loss, using a sample of the negative labels each time.
# loss = tf.reduce_mean(
#   tf.nn.nce_loss(weights=nce_weights,
#                  biases=nce_biases,
#                  labels=train_labels,
#                  inputs=embed,
#                  num_sampled=num_sampled,
#                  num_classes=vocabulary_size))
#
# # We use the SGD optimizer.
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(loss)
#
# for inputs, labels in generate_batch(...):
#   feed_dict = {train_inputs: inputs, train_labels: labels}
#   _, cur_loss = session.run([optimizer, loss], feed_dict=feed_dict)

batch_size = 64
epochs = 110
latent_dim = 256
num_samples = 10000

#stories = get_articles()

input_texts = []
target_texts = []
input_characters = set()
target_characters = set()
i = 0
for story in get_articles():
    input_text = story['fullText']
    for highlight in story['description']:
        target_text = highlight
        # We use "tab" as the "start sequence" character
        # for the targets, and "\n" as "end sequence" character.
        target_text = '\t' + target_text + '\n'
        input_texts.append(input_text)
        target_texts.append(target_text)
        for char in input_text:
            if char not in input_characters:
                input_characters.add(char)
        for char in target_text:
            if char not in target_characters:
                target_characters.add(char)
    i = i + 1
    if i % 10 == 0:
        input_characters_to_file = sorted(list(input_characters))
        target_characters_to_file = sorted(list(target_characters))
        with open('objs2018.pkl', 'wb') as f:
            pickle.dump([input_characters_to_file, target_characters_to_file], f)

input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
with open('objs2018.pkl','w') as f:
    pickle.dump([input_characters, target_characters], f)
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])
print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)

def define_models(n_input, n_output, n_units):
    # define training encoder
    encoder_inputs = Input(shape=(None, n_input))
    encoder = LSTM(n_units, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]
    # define training decoder
    decoder_inputs = Input(shape=(None, n_output))
    decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(n_output, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    # define inference encoder
    encoder_model = Model(encoder_inputs, encoder_states)
    # define inference decoder
    decoder_state_input_h = Input(shape=(n_units,))
    decoder_state_input_c = Input(shape=(n_units,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs,  initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
    # return all models
    return model, encoder_model, decoder_model

model, encoder_model, decoder_model = define_models(input_characters, target_characters, latent_dim)
# Run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit([encoder_input_data, decoder_input_data], decoder_target_data, #todo figure out what these should be
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2)
# Save model
model.save('neural_model.h5')