def __conv_block(x, filters, kernel_size, L2):
    from tensorflow.keras.layers import Conv2D, ReLU
    from tensorflow.keras.initializers import he_normal
    from tensorflow.keras.regularizers import l2

    x = Conv2D(filters, kernel_size=kernel_size, padding="same", strides=(1, 1), kernel_initializer=he_normal(),
               kernel_regularizer=l2(L2))(x)
    x = ReLU()(x)
    return x

def generate_model(input_shape=(416, 416, 3), L2=0.0):
    from tensorflow.keras.layers import Conv2D, Input
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.models import Model

    ####################################################################################################################
    inp = Input(input_shape)
    ####################################################################################################################

    features = __conv_block(x=inp, filters=64, kernel_size=(9, 9), L2=L2)
    features = __conv_block(x=features, filters=32, kernel_size=(3, 3), L2=L2)
    features = Conv2D(3, kernel_size=(5, 5), padding="same", strides=(1, 1), activation="relu", kernel_regularizer=l2(L2))(features)

    ####################################################################################################################
    out = features
    ####################################################################################################################

    model = Model(
        inputs=inp,
        outputs=out,
    )

    return model
