import numpy as np
import keras
from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import matplotlib.pyplot as plt
import pickle

class MyCallback(keras.callbacks.Callback):
    def __init__(self ):
        print()
    def on_epoch_end(self, epoch, logs={}):
        print(self.model)


def load_data( path, num_samples ):
    questions = []
    answers = []
    input_characters = set()
    output_characters = set()
    with open( path, 'r' ) as f:
        lines = f.read().split('\n')
    for i,l in enumerate( lines[:num_samples*2] ):
        if i % 2 == 0:
            answer = l.split( "+++$+++" )[-1]
            answer = '\t' + answer + '\n'
            answers.append( answer )
            for char in answer:
                if char not in output_characters:
                    output_characters.add( char )
        elif i % 2 == 1:
            question = l.split( "+++$+++" )[-1]
            questions.append( question )

            for char in question:
                if char not in input_characters:
                    input_characters.add( char )
    return questions, answers,input_characters, output_characters

def get_model( target_weight_path,checkpoint_path,reverse_target_char_index,model_target_path, train=False, plot=False ):
    # Define an input sequence and process it.
    encoder_inputs = Input(shape=(None, num_encoder_tokens))
    encoder = LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None, num_decoder_tokens))
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                         initial_state=encoder_states)
    decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # Run training

    encoder_model = Model(encoder_inputs, encoder_states)

    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)


    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    if train:
        model_checkpoint = ModelCheckpoint( checkpoint_path )
        blabla_callback = MyCallback()
        #callbacks = [ model_checkpoint, blabla_callback ]
        #callbacks = [ blabla_callback ]
        model.summary()
        history = model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                batch_size=batch_size,
                epochs=epochs,
                validation_split=0.2)
        model.save( model_target_path )
        if plot:
            plot_loss( history )
            # Save model

    else:
        # load new model
        loaded_model= load_model("s2s_200_samples_200_epochs.h5")
        print("Loaded model from disk")
        model_checkpoint = ModelCheckpoint( checkpoint_path )
        callbacks = [ model_checkpoint ]
        loaded_model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
        history = model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                batch_size=batch_size,
                epochs=epochs,
                validation_split=0.2)
        if plot:
            plot_loss( history )
            # Save model
        model.save('s2s.h5')
    #history.append(
     #           new_model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
      #        batch_size=batch_size, epochs=epochs, validation_split=0.2))
        #model.load_weights( target_weight_path )




    return model, encoder_model, decoder_model

def plot_loss( h ):
    plt.plot( h.history['loss'], label='loss' )
    plt.plot( h.history['val_loss'], label='validation loss')
    plt.legend( loc='upper right' )
    plt.show()


def decode_sequence(input_seq, encoder_model, decoder_model ):
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    target_seq[0, 0, target_token_index['\t']] = 1.
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char
        if (sampled_char == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.
        states_value = [h, c]
    return decoded_sentence

def run_test():
    for seq_index in range(100):
        # Take one sequence (part of the training set)
        # for trying out decoding.
        input_seq = encoder_input_data[seq_index: seq_index + 1]
        decoded_sentence = decode_sequence( input_seq, encoder_model, decoder_model )
        print('-')
        print('Input sentence:', input_texts[seq_index])
        print('Decoded sentence:', decoded_sentence)

def sentence_to_vector( sentence, input_token_index ):
    encoder_input_data = np.zeros(
        (1, 1338, 86),
        dtype='float32')
    for t, char in enumerate(sentence):
        encoder_input_data[0, t, input_token_index[char]] = 1.
    return encoder_input_data


def decode_sequence(input_seq ):
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    target_seq[0, 0, target_token_index['\t']] = 1.
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char
        if (sampled_char == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.
        states_value = [h, c]
    return decoded_sentence



















if __name__ == "__main__":
    DATA_PATH = './Data/movie_lines.txt'
    batch_size = 64  # Batch size for training.
    epochs = 1000  # Number of epochs to train for.
    latent_dim = 256  # Latent dimensionality of the encoding space.
    num_samples = 10000 # Number of samples to train on.
    num_samples = 10 # Number of samples to train on.
    model_target_path = "seq2seq_{}_samples_{}_epochs.h5".format( num_samples, epochs)
    TARGET_WEIGHT_PATH = "weights/target_weights_01.hdf5"
    CHECKPOINT_PATH = "weights/target_weights_{epoch:02d}.hdf5"
    input_texts, target_texts,input_characters, target_characters = load_data( DATA_PATH, num_samples )


    input_characters = sorted(list(input_characters))
    target_characters = sorted(list(target_characters))
    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)
    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    print('Number of samples:', len(input_texts))
    print('Number of unique input tokens:', num_encoder_tokens)
    print('Number of unique output tokens:', num_decoder_tokens)
    print('Max sequence length for inputs:', max_encoder_seq_length)
    print('Max sequence length for outputs:', max_decoder_seq_length)

    input_token_index = dict(
        [(char, i) for i, char in enumerate(input_characters)])

    with open( "input_token_index.pkl", 'wb') as f:
        pickle.dump( input_token_index, f, protocol=pickle.HIGHEST_PROTOCOL)

    target_token_index = dict(
        [(char, i) for i, char in enumerate(target_characters)])

    with open( "target_token_index.pkl", 'wb') as f:
        pickle.dump( target_token_index, f, protocol=pickle.HIGHEST_PROTOCOL)

    encoder_input_data = np.zeros(
        (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
        dtype='float32')
    decoder_input_data = np.zeros(
        (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
        dtype='float32')
    decoder_target_data = np.zeros(
        (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
        dtype='float32')


    #for key, value in target_token_index.items() :
        #print (key, value)

    for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
        for t, char in enumerate(input_text):
            encoder_input_data[i, t, input_token_index[char]] = 1.
            for t, char in enumerate(target_text):
                # decoder_target_data is ahead of decoder_input_data by one timestep
                decoder_input_data[i, t, target_token_index[char]] = 1.
                if t > 0:
                    # decoder_target_data will be ahead by one timestep
                    # and will not include the start character.
                    decoder_target_data[i, t - 1, target_token_index[char]] = 1.

    reverse_input_char_index = dict(
        (i, char) for char, i in input_token_index.items())

    reverse_target_char_index = dict(
        (i, char) for char, i in target_token_index.items())



    with open( "reverse_target_char_index.pkl", 'wb') as f:
        pickle.dump( reverse_target_char_index, f, protocol=pickle.HIGHEST_PROTOCOL)

    #model, encoder_model, decoder_model = get_model( TARGET_WEIGHT_PATH, CHECKPOINT_PATH, train=False )
    model, encoder_model, decoder_model = get_model( TARGET_WEIGHT_PATH, CHECKPOINT_PATH,reverse_target_char_index,model_target_path, train=True, plot=True )
    #model, encoder_model, decoder_model = get_model( TARGET_WEIGHT_PATH, CHECKPOINT_PATH,reverse_target_char_index, train=False )


    for seq_index in range(10):
        # Take one sequence (part of the training set)
        # for trying out decoding.
        input_seq = encoder_input_data[seq_index: seq_index + 1]
        decoded_sentence = decode_sequence(input_seq)
        print('-')
        print('Input sentence:', input_texts[seq_index])
        print('Decoded sentence:', decoded_sentence)
