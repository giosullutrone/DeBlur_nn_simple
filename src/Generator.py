from tensorflow.python.keras.utils.data_utils import Sequence

class Generator(Sequence):
    def __init__(self, folder_sharp_images, folder_blurred_images, batch_size, image_exts, shuffle):
        from src.AugmentedImagesUtil import AugmentedImagesUtil
        self.__folder_sharp_images = folder_sharp_images
        self.__folder_blurred_images = folder_blurred_images
        self.__image_files = AugmentedImagesUtil.get_images_file_names_from_folders(folder_sharp_images,
                                                                                    folder_blurred_images,
                                                                                    image_exts=image_exts)
        self.__batch_size = batch_size
        self.__shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(len(self.__image_files) / self.__batch_size)

    def __getitem__(self, index):
        indices = [x for x in range(index*self.__batch_size, (index+1)*self.__batch_size, 1)]
        return self.__data_generation(indices)

    def on_epoch_end(self):
        import numpy as np
        if self.__shuffle:
            np.random.shuffle(self.__image_files)

    def __data_generation(self, indices):
        import numpy as np
        from src.AugmentedImage import AugmentedImage
        from src.DeBlurNetwork import DeBlurNetwork

        inps = [AugmentedImage.image_from_file(self.__folder_blurred_images + self.__image_files[x]) for x in indices]
        outs = [AugmentedImage.image_from_file(self.__folder_sharp_images + self.__image_files[x]) for x in indices]

        inps = DeBlurNetwork.pre_process(np.array(inps))
        outs = DeBlurNetwork.pre_process(np.array(outs))

        return inps, outs
