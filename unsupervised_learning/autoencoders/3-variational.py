#!/usr/bin/env python3
"""Creating a variational autoencoder"""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Function that creates a variational autoencoder

    input_dims is an integer containing the
        dimensions of the model input
    hidden_layers is a list containing the number of nodes
        for each hidden layer in the encoder, respectively
        the hidden layers should be reversed for the decoder
    latent_dims is an integer containing the dimensions
        of the latent space representation

    Return: encoder, decoder, auto

    """
    input_encoder = keras.Input(shape=(input_dims, ))
    input_decoder = keras.Input(shape=(latent_dims, ))

    encoded = input_encoder
    for nodes in hidden_layers:
        encoded = keras.layers.Dense(nodes, activation='relu')(encoded)

    z_mean = keras.layers.Dense(latent_dims, activation=None)(encoded)
    z_log_var = keras.layers.Dense(latent_dims, activation=None)(encoded)

    def sample_z(args):
        """
        Sampling function
        """
        mu, log_var = args
        batch = keras.backend.shape(mu)[0]
        dim = keras.backend.int_shape(mu)[1]
        eps = keras.backend.random_normal(shape=(batch, dim))
        return mu + keras.backend.exp(log_var / 2) * eps

    z = keras.layers.Lambda(sample_z,
                    output_shape=(latent_dims,))([z_mean, z_log_var])

    encoder = keras.Model(inputs=input_encoder,
                  outputs=[z_mean, z_log_var, z])

    decoded = keras.layers.Dense(hidden_layers[-1],
                                 activation='relu')(input_decoder)
    for dec in range(len(hidden_layers) - 2, -1, -1):
        decoded = keras.layers.Dense(hidden_layers[dec],
                                     activation='relu')(decoded)
    last = keras.layers.Dense(input_dims, activation='sigmoid')(decoded)
    decoder = keras.Model(inputs=input_decoder, outputs=last)

    output_auto = decoder(z)
    auto = keras.Model(inputs=input_encoder, outputs=output_auto)

    reconstruction_loss = input_dims * keras.losses.binary_crossentropy(
        input_encoder, output_auto)
    kl_loss = -0.5 * keras.backend.sum(
        1 + z_log_var - keras.backend.square(z_mean) - keras.backend.exp(
            z_log_var), axis=-1)
    auto.add_loss(keras.backend.mean(reconstruction_loss + kl_loss))
    auto.compile(optimizer='adam')

    return encoder, decoder, auto
