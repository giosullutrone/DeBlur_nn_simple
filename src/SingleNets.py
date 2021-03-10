def __conv_block(x, filters, kernel_size, L2):
    from tensorflow.keras.layers import Conv2D, ReLU
    from tensorflow.keras.initializers import he_normal
    from tensorflow.keras.regularizers import l2

    x = Conv2D(filters, kernel_size=kernel_size, padding="same", strides=(1, 1), kernel_initializer=he_normal(seed=42),
               kernel_regularizer=l2(L2))(x)
    x = ReLU()(x)
    return x

def __conv_block_grouped(x, filters, kernel_sizes, L2):
    from tensorflow.keras.layers import Conv2D, ReLU, Lambda, concatenate
    from tensorflow.keras.initializers import he_normal
    from tensorflow.keras.regularizers import l2
    import tensorflow.keras.backend as K

    x_shape = list(K.int_shape(x))
    x_filters_per_group = int(x_shape[-1] / len(kernel_sizes))
    x_shape_lambda = x_shape[:-1] + [x_filters_per_group]

    x_groups = []
    for index, kernel_size in enumerate(kernel_sizes):
        x_group = Lambda(lambda z: z[..., index * x_filters_per_group: (index+1) * x_filters_per_group],
                         output_shape=x_shape_lambda)(x)

        x_group = Conv2D(int(filters / len(kernel_sizes)), kernel_size, padding="same", kernel_initializer=he_normal(seed=42),
                         kernel_regularizer=l2(L2))(x_group)
        x_group = ReLU()(x_group)

        x_groups.append(x_group)
    return concatenate(inputs=x_groups)

def generate_simple_conv_model(input_shape, L2):
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

def generate_grouped_conv_model(input_shape, kernel_sizes, L2):
    from tensorflow.keras.layers import Conv2D, Input
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.models import Model

    ####################################################################################################################
    inp = Input(input_shape)
    ####################################################################################################################
    features = __conv_block(x=inp, filters=32, kernel_size=(9, 9), L2=L2)
    features = __conv_block_grouped(x=features, filters=64, kernel_sizes=kernel_sizes, L2=L2)
    features = __conv_block_grouped(x=features, filters=32, kernel_sizes=kernel_sizes, L2=L2)
    features = Conv2D(3, kernel_size=(5, 5), padding="same", strides=(1, 1), activation="relu", kernel_regularizer=l2(L2))(features)

    ####################################################################################################################
    out = features
    ####################################################################################################################

    model = Model(
        inputs=inp,
        outputs=out,
    )

    return model
