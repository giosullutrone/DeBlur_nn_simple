class DeBlurAdversarialNet:
    def __init__(self, input_shape):
        self.__model_generator, self.__model_discriminator, self.__model_gan = self.__init_model(input_shape)

        self.__model_generator.summary()
        self.__model_discriminator.summary()
        self.__model_gan.summary()

    @staticmethod
    def __init_model(input_shape):
        from src.AdversarialNets import generate_simple_conv_generator, generate_simple_conv_discriminator, generate_gan_from_models

        gen = generate_simple_conv_generator(input_shape=input_shape, L2=0.0)
        dis = generate_simple_conv_discriminator(input_shape=input_shape, L2=0.0)

        return gen, dis, generate_gan_from_models(generator=gen, discriminator=dis)

    def train_model(self, epochs,
                    folder_weights_save,
                    generator_discriminator,
                    generator_gan,
                    generator_validation=None,
                    path_weights_load=None):
        from tensorflow.keras.callbacks import ModelCheckpoint

        generator_discriminator.set_model_generator(self.__model_generator)

        ################################################################################################################
        # Compile models
        ################################################################################################################
        lr = 0.001
        self.__compile_models(lr=lr)
        ################################################################################################################

        ################################################################################################################
        # Load previous GAN weights if needed
        ################################################################################################################
        if path_weights_load is not None:
            self.load_weights(path_weights_load)
        ################################################################################################################

        ################################################################################################################
        # Get a batch of validation data if generator provided
        ################################################################################################################
        validation_data_x = None
        validation_data_y = None
        if generator_validation is not None:
            x, y = generator_validation.__getitem__(0)
            validation_data_x = x
            validation_data_y = y
        ################################################################################################################
        checkpoint = [ModelCheckpoint(folder_weights_save + "deblur.h5",
                                      monitor="val_gen_out_loss",
                                      verbose=1,
                                      save_best_only=True,
                                      save_weights_only=False)
                      ]

        reduce_lr_each_n_epochs = 10
        reduce_lr_factor = 0.5
        for i in range(epochs):
            print("Epoch: " + str(i))

            ############################################################################################################
            # Fit Discriminator
            ############################################################################################################
            print("Training discriminator...")
            self.__model_discriminator.fit(x=generator_discriminator,
                                           workers=0,
                                           epochs=1,
                                           verbose=1)
            ############################################################################################################

            ############################################################################################################
            # Fit GAN
            ############################################################################################################
            print("Training GAN...")
            self.__model_gan.fit(x=generator_gan,
                                 callbacks=checkpoint,
                                 epochs=1,
                                 validation_data=(validation_data_x, validation_data_y),
                                 verbose=1)

            if i % reduce_lr_each_n_epochs == 0 and i != 0:
                lr_old = lr
                lr = lr * reduce_lr_factor
                print(f"Reducing lr from {lr_old} to {lr}.")
                self.__compile_models(lr=lr)
            ############################################################################################################

    def __compile_models(self, lr: float):
        ################################################################################################################
        # Get loss, optimizer and then compile model
        # Note: compile discriminator with trainable = True (to save it for training)
        #       compile GAN with discriminator trainable = False (to save it for training)
        ################################################################################################################
        self.__set_discriminator_trainable(trainable=True)
        self.__model_discriminator.compile(DeBlurAdversarialNet.__optimizer(lr=lr),
                                           loss=DeBlurAdversarialNet.__loss_function_discriminator)

        self.__set_discriminator_trainable(trainable=False)
        self.__model_gan.compile(DeBlurAdversarialNet.__optimizer(lr=lr),
                                 loss=[DeBlurAdversarialNet.__loss_function_generator,
                                       DeBlurAdversarialNet.__loss_function_discriminator],
                                 loss_weights=[1.0, 1e-3])

    def __set_discriminator_trainable(self, trainable: bool):
        for layer in self.__model_discriminator.layers:
            layer.trainable = trainable

    def load_weights(self, path_weights_load):
        try:
            self.__model_gan.load_weights(path_weights_load)
        except Exception as e:
            print(e)

    def __predict_from_image(self, image):
        """Returns a sharp image from a blurred image"""
        import numpy as np
        prediction = self.__model_gan.predict(DeBlurAdversarialNet.pre_process(np.expand_dims(image, axis=0)))[0]
        return DeBlurAdversarialNet.post_process(prediction[0])

    def __predict_from_images(self, images):
        """Returns sharp images from a list of blurred images"""
        import numpy as np
        prediction = self.__model_gan.predict(DeBlurAdversarialNet.pre_process(np.array(images)))[0]
        return DeBlurAdversarialNet.post_process(prediction)

    def predict_from_image(self, image, section_size):
        from src.AugmentedImage import AugmentedImage
        sections_annotated = AugmentedImage.get_sections_annotated(image, section_size)

        images_to_predict = []
        for section in sections_annotated:
            __, __, image_to_predict = section
            images_to_predict.append(image_to_predict)

        predictions = self.__predict_from_images(images_to_predict)

        sections_predicted = []
        for section, prediction in zip(sections_annotated, predictions):
            x, y, __ = section
            sections_predicted.append((x, y, prediction))

        return AugmentedImage.image_from_sections_annotated(image, sections_predicted)

    @staticmethod
    def __optimizer(lr):
        from tensorflow.keras.optimizers import Adam
        return Adam(lr=lr)

    @staticmethod
    def __loss_function_discriminator(y_true, y_pred):
        from tensorflow.keras.losses import binary_crossentropy
        return binary_crossentropy(y_true, y_pred)

    @staticmethod
    def __loss_function_generator(y_true, y_pred):
        from tensorflow.keras.losses import mse
        return mse(y_true, y_pred)

    @staticmethod
    def pre_process(x):
        return (x / 255.0) - 0.0

    @staticmethod
    def post_process(x):
        return (x + 0.0) * 255.0
