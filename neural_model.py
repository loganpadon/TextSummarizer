import pickle
from numpy import array
from numpy import append
from numpy import zeros
from keras import Input, Model
from keras.layers import Dense
from keras.layers import LSTM
from string import ascii_lowercase

batch_size = 64
epochs = 110
latent_dim = 256
num_samples = 10000

# input_texts = []
# target_texts = []
# input_characters = set()
# target_characters = set()
# i = 0
# for story in get_articles(year=2017):
#     input_text = story['fullText']
#     target_text = story['description']
#     # We use "tab" as the "start sequence" character
#     # for the targets, and "\n" as "end sequence" character.
#     target_text = '\t' + target_text + '\n'
#     input_texts.append(input_text)
#     target_texts.append(target_text)
#     for char in input_text:
#         if char not in input_characters:
#             input_characters.add(char)
#     for char in target_text:
#         if char not in target_characters:
#             target_characters.add(char)
#     i = i + 1
#     if i % 10 == 0:
#         input_characters_to_file = sorted(list(input_characters))
#         target_characters_to_file = sorted(list(target_characters))
#         with open('objs2017.pkl', 'wb') as f:
#             pickle.dump([input_characters_to_file, target_characters_to_file, input_texts, target_texts], f)
#
#     if i == 1000:
#         break

pickle_in = open("objs2017.pkl", "rb")
asdf = pickle.load(pickle_in)
input_characters = asdf[0]
target_characters = asdf[1]
input_texts = asdf[2]
target_texts = asdf[3]
asdf = None
input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
# with open('objs2017.pkl','wb') as f:
#     pickle.dump([input_characters, target_characters], f)
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


def string_vectorizer(strng, alphabet=ascii_lowercase):
    # vector = array([0 if char != letter else 1 for char in alphabet
    #               for letter in strng])
    vector = zeros(shape=(len(strng), len(alphabet)))
    i = 0
    for x in strng:
        j = 0
        for y in alphabet:
            j += 1
            if x == y:
                vector[i, j] = 1
        i += 1
    return vector


model, encoder_model, decoder_model = define_models(num_encoder_tokens, num_decoder_tokens, latent_dim)
# Run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
itVector = zeros(shape=(len(input_texts), 100, 26), dtype=int)
for input_text in input_texts:
    i = 0
    input_text = input_text.lower()
    vector = string_vectorizer(input_text)
    itVector[i] = vector
    i += 1

ttVector = []
for target_text in target_texts:
    target_text = target_text.lower()
    vector = string_vectorizer(target_text)
    append(ttVector, vector)

model.fit([array(itVector), array(ttVector)], ttVector, #todo figure out what these should be
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)
# Save model
model.save('neural_model.h5')
