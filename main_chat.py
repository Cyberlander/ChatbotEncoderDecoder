import numpy as np
from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import matplotlib.pyplot as plt
import pickle

def get_model( MODEL_PATH, num_encoder_tokens,num_decoder_tokens ):
    # Define an input sequence and process it.
    #encoder_inputs = Input(shape=(None, num_encoder_tokens))
    #encoder = LSTM(latent_dim, return_state=True)
    #encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    # We discard `encoder_outputs` and only keep the states.
    #encoder_states = [state_h, state_c]

    ## Set up the decoder, using `encoder_states` as initial state.
    #decoder_inputs = Input(shape=(None, num_decoder_tokens))
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    #decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    #decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                        # initial_state=encoder_states)
    #decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    #decoder_outputs = decoder_dense(decoder_outputs)


    model = load_model( MODEL_PATH )

    encoder_inputs = Input(shape=(None, num_encoder_tokens))
    decoder_inputs = Input(shape=(None, num_decoder_tokens))

    encoder = model.get_layer( "lstm_1" )
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]
    encoder_model = Model(encoder_inputs, encoder_states)


    decoder_lstm = model.get_layer( "lstm_2" )
    decoder_dense = model.get_layer( "dense_1" )

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
    return model, encoder_model, decoder_model










def sentence_to_vector( sentence, input_token_index, num_encoder_tokens ):
    encoder_input_data = np.zeros(
        (1, 1338, num_encoder_tokens),
        dtype='float32')
    for t, char in enumerate(sentence):
        encoder_input_data[0, t, input_token_index[char]] = 1.
    return encoder_input_data


def decode_sequence(input_seq, encoder_model, decoder_model, reverse_target_char_index ):
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

    num_encoder_tokens = 59
    num_decoder_tokens = 64
    latent_dim = 256

    max_encoder_seq_length = 219
    max_decoder_seq_length = 165


    MODEL_PATH = 's2s.h5'
    #MODEL_PATH = 'models/max_30epochs_1805012.h5'


    model, encoder_model, decoder_model = get_model( MODEL_PATH,num_encoder_tokens,num_decoder_tokens )
    with open( "input_token_index.pkl", 'rb') as handle:
        input_token_index = pickle.load(handle)

    with open( "reverse_target_char_index.pkl", 'rb') as f:
        reverse_target_char_index = pickle.load(f)

    with open( "target_token_index.pkl", 'rb') as f:
        target_token_index = pickle.load(f)

    #for key, value in target_token_index.items() :
    #    print (key, value)

    chat = True
    while chat:
        i = input("you: ")
        sentence_vector = sentence_to_vector( i, input_token_index,num_encoder_tokens )
        answer = decode_sequence(sentence_vector, encoder_model, decoder_model, reverse_target_char_index )

        #TODO to seq
        print("bot: {}".format( answer ))
        if i == "exit":
            chat=False
