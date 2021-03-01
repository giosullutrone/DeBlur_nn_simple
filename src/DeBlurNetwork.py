class DeBlurNetwork:
    def __init__(self, model):
        self.__model = model

    def get_model(self):
        return self.__model

    def set_model(self, model):
        self.__model = model

    def train_model(self, epochs, steps_per_epoch, generator, folder_weights_save, generator_validation=None, folder_weights_load=None, do_checkpoint=True, optimizer=None, loss_function=None):
        from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
        import numpy as np

        loss_function = DeBlurNetwork.loss_function if loss_function is None else loss_function
        optimizer = DeBlurNetwork.optimizer() if optimizer is None else optimizer

        self.__model.compile(optimizer, loss=loss_function)

        if folder_weights_load is not None:
            self.load_checkpoint(folder_weights_load)

        checkpoint = [ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            verbose=1,
            mode="min",
        )]

        if do_checkpoint:
            checkpoint += [ModelCheckpoint(folder_weights_save + "deblur_checkpoint.h5", monitor="val_loss", verbose=1,
                                           save_best_only=True, save_weights_only=False),
                           ModelCheckpoint(folder_weights_save + "deblur.h5", monitor="val_loss", verbose=1,
                                           save_best_only=True, save_weights_only=True)]

        validation_data_x = None
        validation_data_y = None
        if generator_validation is not None:
            x, y = next(generator_validation)
            validation_data_x = x
            validation_data_y = y

        self.__model.fit_generator(generator=generator,
                                   steps_per_epoch=steps_per_epoch,
                                   epochs=epochs,
                                   callbacks=checkpoint,
                                   validation_data=(validation_data_x, validation_data_y),
                                   verbose=1)

        self.__model.save_weights(filepath=folder_weights_save + "deblur.h5")
        self.__model.save(filepath=folder_weights_save + "deblur_checkpoint.h5")

    def load_weights(self, folder_weights_load):
        self.__model.load_weights(folder_weights_load + "deblur.h5", by_name=False)

    def load_checkpoint(self, folder_weights_load):
        from tensorflow.keras.models import load_model
        try:
            self.__model = load_model(folder_weights_load + "deblur_checkpoint.h5",
                                      custom_objects={"loss_function": DeBlurNetwork.loss_function})
        except Exception as e:
            #print("Could not load the weights: " + folder_weights_load + " not found, skipping...")
            print(e)

    def predict_from_image_full(self, image, section_size=(416, 416)):
        """Returns the reconstructed image from predictions"""
        from src.AugmentedImage import AugmentedImage
        import numpy as np

        aug_image = AugmentedImage(image)

        sections = aug_image.get_sections_annotated(section_size)

        to_predict = []
        for section in sections:
            _, _, image_section = section
            to_predict.append(image_section)

        predictions = self.__model.predict(DeBlurNetwork.pre_process(np.array(to_predict)))

        sections_predicted = []
        for section, prediction in zip(sections, predictions):
            x, y, _ = section
            image_predicted = prediction
            sections_predicted.append((x, y, DeBlurNetwork.post_process(image_predicted)))

        return aug_image.image_from_sections_annotated(sections_predicted)

    def predict_from_image(self, image):
        """Returns the prediction from the image"""
        from src.AugmentedImage import AugmentedImage
        import numpy as np

        aug_image = AugmentedImage(image)

        prediction = self.__model.predict(DeBlurNetwork.pre_process(np.expand_dims(aug_image.get_image(), axis=0)))[0]

        return DeBlurNetwork.post_process(prediction)

    @staticmethod
    def optimizer(lr=0.001, amsgrad=False):
        from tensorflow.keras.optimizers import Adam
        return Adam(lr=lr, amsgrad=amsgrad)

    @staticmethod
    def loss_function(y_true, y_pred):
        import tensorflow.keras.backend as K
        from tensorflow.keras.losses import mse, mae

        vertical = K.constant([[0.0, 1.0, 0.0],
                               [0.0, 1.0, 0.0],
                               [0.0, 1.0, 0.0]])
        horizontal = K.constant([[0.0, 0.0, 0.0],
                                 [1.0, 1.0, 1.0],
                                 [0.0, 0.0, 0.0]])
        oblique_0 = K.constant([[1.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0],
                                [0.0, 0.0, 1.0]])
        oblique_1 = K.constant([[0.0, 0.0, 1.0],
                                [0.0, 1.0, 0.0],
                                [1.0, 0.0, 0.0]])

        return mse(y_true, y_pred)

    @staticmethod
    def pre_process(x):
        return (x / 255.0) - 0.0

    @staticmethod
    def post_process(x):
        return (x + 0.0) * 255.0

    @staticmethod
    def generator(folder_sharp_images, folder_blurred_images, batch_size=64, section_size=(416, 416, 3), image_ext=(".jpg", ".png", ".jpeg")):
        import numpy as np
        import random
        from src.AugmentedImagesUtil import AugmentedImagesUtil
        from src.AugmentedImage import AugmentedImage

        image_files = AugmentedImagesUtil.get_images_file_names_from_folders(folder_sharp_images, folder_blurred_images, image_exts=image_ext)

        while True:
            batch_files = random.choices(image_files, k=batch_size)
            batch_input = []
            batch_output = []

            for batch_file in batch_files:
                image_sharp_file, image_blurred_file = batch_file

                aug_sharp = AugmentedImage.from_file(folder_sharp_images + image_sharp_file, grayscale=False)
                aug_blurred = AugmentedImage.from_file(folder_blurred_images + image_blurred_file, grayscale=False)

                out = aug_sharp.get_image()
                inp = aug_blurred.get_image()

                if np.isnan(np.sum(inp)) or np.isnan(np.sum(out)):
                    print("Found an NaN in input and/or output, skipping the file...")
                    continue

                inp = DeBlurNetwork.pre_process(inp)
                out = DeBlurNetwork.pre_process(out)
                batch_input += [inp]
                batch_output += [out]

            batch_x = np.array(batch_input)
            batch_y = np.array(batch_output)

            yield batch_x, batch_y
