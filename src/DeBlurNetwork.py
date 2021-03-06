class DeBlurNetwork:
    def __init__(self, input_shape):
        from src.Net import generate_model
        self.__model = generate_model(input_shape=input_shape, L2=0.0)

    def get_model(self):
        return self.__model

    def train_model(self, epochs, steps_per_epoch,
                    generator, folder_weights_save,
                    generator_validation=None, path_weights_load=None):
        from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

        ################################################################################################################
        # Get loss, optimizer and then compile model
        ################################################################################################################
        loss_function = DeBlurNetwork.__loss_function
        optimizer = DeBlurNetwork.__optimizer(lr=0.001)
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
            x, y = next(generator_validation)
            validation_data_x = x
            validation_data_y = y
        ################################################################################################################

        ################################################################################################################
        # Fit and save model
        ################################################################################################################
        self.__model.fit(x=generator,
                         steps_per_epoch=steps_per_epoch,
                         epochs=epochs,
                         callbacks=checkpoint,
                         validation_data=(validation_data_x, validation_data_y),
                         verbose=1)
        self.__model.save(filepath=folder_weights_save + "deblur.h5")
        ################################################################################################################

    def load_weights(self, path_weights_load):
        try:
            self.__model.load_weights(path_weights_load)
        except Exception as e:
            print(e)

    def load_checkpoint(self, path_weights_load):
        from tensorflow.keras.models import load_model
        try:
            self.__model = load_model(path_weights_load, custom_objects={"__loss_function": DeBlurNetwork.__loss_function})
        except Exception as e:
            print(e)

    def __predict_from_image(self, image):
        """Returns a sharp image from a blurred image"""
        import numpy as np
        prediction = self.__model.predict(DeBlurNetwork.__pre_process(np.expand_dims(image, axis=0)))
        return DeBlurNetwork.__post_process(prediction[0])

    def __predict_from_images(self, images):
        """Returns sharp images from a list of blurred images"""
        import numpy as np
        prediction = self.__model.predict(DeBlurNetwork.__pre_process(np.array(images)))
        return DeBlurNetwork.__post_process(prediction)

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
    def __pre_process(x):
        return (x / 255.0) - 0.0

    @staticmethod
    def __post_process(x):
        return (x + 0.0) * 255.0

    @staticmethod
    def get_number_of_steps(folder_images, batch_size, image_exts):
        from src.AugmentedImagesUtil import AugmentedImagesUtil
        return int(len(AugmentedImagesUtil.get_images_file_names_from_folder(folder_images, image_exts)) / batch_size)

    @staticmethod
    def generator(folder_sharp_images, folder_blurred_images, batch_size, section_size, image_exts):
        import numpy as np
        import random
        from src.AugmentedImagesUtil import AugmentedImagesUtil
        from src.AugmentedImage import AugmentedImage

        image_files = AugmentedImagesUtil.get_images_file_names_from_folders(folder_sharp_images, folder_blurred_images, image_exts=image_exts)

        while True:
            batch_files = random.choices(image_files, k=batch_size)
            batch_input = []
            batch_output = []

            for batch_file in batch_files:
                image_sharp_file, image_blurred_file = batch_file

                aug_sharp = AugmentedImage.image_from_file(folder_sharp_images + image_sharp_file, grayscale=False)
                aug_blurred = AugmentedImage.image_from_file(folder_blurred_images + image_blurred_file, grayscale=False)

                out = aug_sharp
                inp = aug_blurred

                if np.isnan(np.sum(inp)) or np.isnan(np.sum(out)):
                    print("Found an NaN in input and/or output, skipping file...")
                    continue

                inp = DeBlurNetwork.__pre_process(inp)
                out = DeBlurNetwork.__pre_process(out)
                batch_input += [inp]
                batch_output += [out]

            batch_x = np.array(batch_input)
            batch_y = np.array(batch_output)

            yield batch_x, batch_y
