class DeBlurSingleNet:
    def __init__(self, input_shape, model_type):
        self.__model = self.__init_model(input_shape, model_type)
        self.__model.summary()

    @staticmethod
    def __init_model(input_shape, model_type):
        from src.SingleNets import generate_simple_conv_model, generate_grouped_conv_model

        if model_type == "s":
            return generate_simple_conv_model(input_shape=input_shape, L2=0.0)
        elif model_type == "g":
            return generate_grouped_conv_model(input_shape=input_shape, kernel_sizes=(3, 5, 7, 9), L2=0.0)

    def train_model(self, epochs,
                    generator,
                    folder_weights_save,
                    generator_validation=None,
                    path_weights_load=None):
        from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

        ################################################################################################################
        # Get loss, optimizer and then compile model
        ################################################################################################################
        loss_function = DeBlurSingleNet.__loss_function
        optimizer = DeBlurSingleNet.__optimizer(lr=0.001)
        self.__model.compile(optimizer, loss=loss_function)
        ################################################################################################################

        ################################################################################################################
        # Load previous checkpoint if needed
        ################################################################################################################
        if path_weights_load is not None:
            self.load_checkpoint(path_weights_load)
        ################################################################################################################

        ################################################################################################################
        # Add checkpoints and reduceLrOnPlateau
        ################################################################################################################
        checkpoint = [ReduceLROnPlateau(monitor="val_loss",
                                        factor=0.5,
                                        patience=5,
                                        verbose=1,
                                        mode="min"),
                      ModelCheckpoint(folder_weights_save + "deblur.h5",
                                      monitor="val_loss",
                                      verbose=1,
                                      save_best_only=True,
                                      save_weights_only=False)
                      ]
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

        ################################################################################################################
        # Fit and save model
        ################################################################################################################
        self.__model.fit(x=generator,
                         epochs=epochs,
                         callbacks=checkpoint,
                         validation_data=(validation_data_x, validation_data_y),
                         verbose=1)
        ################################################################################################################

    def load_weights(self, path_weights_load):
        try:
            self.__model.load_weights(path_weights_load)
        except Exception as e:
            print(e)

    def load_checkpoint(self, path_weights_load):
        from tensorflow.keras.models import load_model
        try:
            self.__model = load_model(path_weights_load, custom_objects={"__loss_function": DeBlurSingleNet.__loss_function})
        except Exception as e:
            print(e)

    def __predict_from_image(self, image):
        """Returns a sharp image from a blurred image"""
        import numpy as np
        prediction = self.__model.predict(DeBlurSingleNet.pre_process(np.expand_dims(image, axis=0)))
        return DeBlurSingleNet.post_process(prediction[0])

    def __predict_from_images(self, images):
        """Returns sharp images from a list of blurred images"""
        import numpy as np
        prediction = self.__model.predict(DeBlurSingleNet.pre_process(np.array(images)))
        return DeBlurSingleNet.post_process(prediction)

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
    def __loss_function(y_true, y_pred):
        from tensorflow.keras.losses import mse
        return mse(y_true, y_pred)

    @staticmethod
    def pre_process(x):
        return (x / 255.0) - 0.0

    @staticmethod
    def post_process(x):
        return (x + 0.0) * 255.0
