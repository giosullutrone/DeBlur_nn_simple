def __conv_block(x, filters, strides, kernel_size, L2):
    from tensorflow.keras.layers import Conv2D, ReLU
    from tensorflow.keras.initializers import he_normal
    from tensorflow.keras.regularizers import l2

    x = Conv2D(filters, kernel_size=kernel_size, padding="same", strides=strides, kernel_initializer=he_normal(seed=42),
               kernel_regularizer=l2(L2))(x)
    x = ReLU()(x)
    return x

def generate_simple_conv_generator(input_shape, L2):
    from tensorflow.keras.layers import Conv2D, Input
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.models import Model

    ####################################################################################################################
    inp = Input(input_shape)
    ####################################################################################################################

    features = __conv_block(x=inp, filters=64, kernel_size=(9, 9), strides=(1, 1), L2=L2)
    features = __conv_block(x=features, filters=32, kernel_size=(3, 3), strides=(1, 1), L2=L2)
    features = Conv2D(3, kernel_size=(5, 5), padding="same", strides=(1, 1), activation="sigmoid", kernel_regularizer=l2(L2))(features)

    ####################################################################################################################
    out = features
    ####################################################################################################################

    model = Model(
        inputs=inp,
        outputs=out,
    )

    return model

def generate_simple_conv_discriminator(input_shape, L2):
    from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.initializers import glorot_normal
    from tensorflow.keras.models import Model

    ####################################################################################################################
    inp = Input(input_shape)
    ####################################################################################################################

    features = __conv_block(x=inp, filters=64, kernel_size=(3, 3), strides=(2, 2), L2=L2)
    features = __conv_block(x=features, filters=64, kernel_size=(3, 3), strides=(2, 2), L2=L2)
    features = __conv_block(x=features, filters=64, kernel_size=(3, 3), strides=(2, 2), L2=L2)
    features = __conv_block(x=features, filters=64, kernel_size=(3, 3), strides=(2, 2), L2=L2)
    features = __conv_block(x=features, filters=64, kernel_size=(3, 3), strides=(2, 2), L2=L2)

    ####################################################################################################################
    out = GlobalAveragePooling2D()(features)
    out = Dense(1, activation="sigmoid", kernel_initializer=glorot_normal(seed=42), kernel_regularizer=l2(L2))(out)
    ####################################################################################################################

    model = Model(
        inputs=inp,
        outputs=out,
    )

    return model

def get_gan_from_models(generator, discriminator):
    from tensorflow.keras.models import Model
    gan = discriminator(generator.outputs)

    return Model(
        inputs=generator.inputs,
        outputs=[generator.outputs, gan],
    )
