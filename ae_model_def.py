import tensorflow as tf
from tensorflow.keras import layers


class Encoder_T(layers.Layer):
    """Maps patch-seq transcriptomic profiles to latent space"""

    def __init__(self,
                 dropout_rate=0.5,
                 latent_dim=3,
                 intermediate_dim=50,
                 training=True,
                 name='Encoder_T',
                 dtype=tf.float32,
                 **kwargs):
        """ Encoder for transcriptomic data
        Args:
            dropout_rate: Dropout probability for dropout layer.
            latent_dim: latent space dimenionality.
            intermediate_dim: Number of units in hidden layers
            name: 'Encoder_T'
        """
        super(Encoder_T, self).__init__(name=name, **kwargs)
        self.drp = layers.Dropout(rate=dropout_rate)
        self.fc0 = layers.Dense(intermediate_dim, activation='relu', name=name+'fc0')
        self.fc1 = layers.Dense(intermediate_dim, activation='relu', name=name+'fc1')
        self.fc2 = layers.Dense(intermediate_dim, activation='relu', name=name+'fc2')
        self.fc3 = layers.Dense(intermediate_dim, activation='relu', name=name+'fc3')
        self.fc4 = layers.Dense(latent_dim, activation='linear', name=name+'fc4')
        self.bn = layers.BatchNormalization(scale=False, center=False, epsilon=1e-10, momentum=0.0, name=name+'BN')
        return

    def call(self, inputs, training=True):
        x = self.drp(inputs, training=training)
        x = self.fc0(x, training=training)
        x = self.fc1(x, training=training)
        x = self.fc2(x, training=training)
        x = self.fc3(x, training=training)
        x = self.fc4(x, training=training)
        z = self.bn(x, training=training)
        return z


class Decoder_T(layers.Layer):
    """Reconstructs gene profile from latent space position"""

    def __init__(self,
                 output_dim,
                 intermediate_dim=50,
                 name='Decoder_T',
                 dtype=tf.float32,
                 **kwargs):
        """Decoder for transcriptomic data
        Args:
            output_dim: Same as input dimensionality if using as an autoencoder.
            intermediate_dim: Number of units in hidden layers
            training: boolean value to indicate model operation mode
            name: 'Decoder_T'
        """
        super(Decoder_T, self).__init__(name=name, **kwargs)
        self.fc0 = layers.Dense(intermediate_dim, activation='relu', name='fc0')
        self.fc1 = layers.Dense(intermediate_dim, activation='relu', name='fc1')
        self.fc2 = layers.Dense(intermediate_dim, activation='relu', name='fc2')
        self.fc3 = layers.Dense(intermediate_dim, activation='relu', name='fc3')
        self.Xout = layers.Dense(output_dim, activation='relu', name='Xout')
        return

    def call(self, inputs, training=True):
        x = self.fc0(inputs, training=training)
        x = self.fc1(x, training=training)
        x = self.fc2(x, training=training)
        x = self.fc3(x, training=training)
        x = self.Xout(x)
        return x


class Encoder_E(layers.Layer):
    """Maps patch-seq electrophysiology profiles to latent space"""

    def __init__(self,
                 gaussian_noise_sd=0.05,
                 dropout_rate=0.1,
                 latent_dim=3,
                 intermediate_dim=40,
                 name='Encoder_E',
                 dtype=tf.float32,
                 **kwargs):
        """
        Initializes the Encoder for electrophysiology data.
        Args:
            gaussian_noise_sd: Noise addition in training mode to E features
            dropout_rate: Dropout probability for dropout layer.
            latent_dim: latent space dimenionality.
            intermediate_dim: Number of units in hidden layers
        """
        super(Encoder_E, self).__init__(name=name, **kwargs)
        self.gnoise = layers.GaussianNoise(stddev=gaussian_noise_sd)
        self.drp = layers.Dropout(rate=dropout_rate)
        self.fc0 = layers.Dense(intermediate_dim, activation='relu', name=name+'fc0')
        self.fc1 = layers.Dense(intermediate_dim, activation='relu', name=name+'fc1')
        self.fc2 = layers.Dense(intermediate_dim, activation='relu', name=name+'fc2')
        self.fc3 = layers.Dense(intermediate_dim, activation='relu', name=name+'fc3')
        self.fc4 = layers.Dense(latent_dim, activation='linear', name=name+'fc4')
        self.bn = layers.BatchNormalization(scale=False, center=False, epsilon=1e-10, momentum=0.0, name=name+'BN')
        return

    def call(self, inputs, training=True):
        x = self.gnoise(inputs, training=training)
        x = self.drp(inputs, training=training)
        x = self.fc0(x, training=training)
        x = self.fc1(x, training=training)
        x = self.fc2(x, training=training)
        x = self.fc3(x, training=training)
        x = self.fc4(x, training=training)
        z = self.bn(x, training=training)
        return z

class Decoder_E(layers.Layer):
    """Reconstructs gene profile from latent space position"""

    def __init__(self,
                 output_dim,
                 intermediate_dim=40,
                 name='Decoder_E',
                 dtype=tf.float32,
                 **kwargs):
        """
        Initializes the Encoder for electrophysiology data.
        Args:
            output_dim: Should be same as input dim if using as an autoencoder
            intermediate_dim: Number of units in hidden layers
            training: boolean value to indicate model operation mode
        """
        super(Decoder_E, self).__init__(name=name, **kwargs)
        self.fc0  = layers.Dense(intermediate_dim, activation='relu',name=name+'fc0')
        self.fc1  = layers.Dense(intermediate_dim, activation='relu',name=name+'fc1')
        self.fc2  = layers.Dense(intermediate_dim, activation='relu',name=name+'fc2')
        self.fc3  = layers.Dense(intermediate_dim, activation='relu',name=name+'fc3')
        self.Xout = layers.Dense(output_dim, activation='linear',name=name+'Xout')
        return

    def call(self, inputs, training=True):
        x = self.fc0(inputs, training=training)
        x = self.fc1(x, training=training)
        x = self.fc2(x, training=training)
        x = self.fc3(x, training=training)
        x = self.Xout(x, training=training)
        return x


class Model_TE(tf.keras.Model):
    """Combine two AE agents"""

    def __init__(self,
               T_output_dim,
               E_output_dim,
               T_intermediate_dim=50,
               E_intermediate_dim=40,
               T_dropout=0.5,
               E_gnoise_sd=0.5,
               E_dropout=0.1,
               latent_dim=3,
               name='TE',
               **kwargs):
        """
        Encoder for transcriptomic data
        Args:
            T_output_dim: Number of genes in T data
            E_output_dim: Number of features in E data
            T_intermediate_dim: hidden layer dims for T model
            E_intermediate_dim: hidden layer dims for E model
            T_dropout: dropout for T data
            E_gnoise_sd: gaussian noise std for E data
            E_dropout: dropout for E data
            latent_dim: dim for representations
            name: TE
        """
        super(Model_TE, self).__init__(name=name, **kwargs)
        self.encoder_T = Encoder_T(dropout_rate=0.5,latent_dim=latent_dim, intermediate_dim=T_intermediate_dim, name='Encoder_T')
        self.encoder_E = Encoder_E(gaussian_noise_sd=0.1, dropout_rate=0.1, latent_dim=3, intermediate_dim=E_intermediate_dim, name='Encoder_E')
        
        self.decoder_T = Decoder_T(output_dim=T_output_dim, intermediate_dim=T_intermediate_dim, name='Decoder_T')
        self.decoder_E = Decoder_E(output_dim=E_output_dim, intermediate_dim=E_intermediate_dim, name='Decoder_E')

    def call(self, inputs, training):
        zT = self.encoder_T(inputs[0],training=training)
        zE = self.encoder_E(inputs[1],training=training)
        XrT = self.decoder_T(zT,training=training)
        XrE = self.decoder_E(zE,training=training)
        return zT,zE,XrT,XrE
