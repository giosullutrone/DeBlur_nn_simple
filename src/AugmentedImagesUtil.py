class AugmentedImagesUtil:
    @staticmethod
    def get_images_file_names_from_folder(folder_images, image_exts):
        """Returns the file names of images in the provided folder"""
        import os
        return [x for x in os.listdir(folder_images) if x.endswith(image_exts)]

    @staticmethod
    def get_images_file_names_from_folders(folder_sharp, folder_blurred, image_exts):
        """Returns the file names of images that exists in both folders as a list of tuples"""
        import os

        image_sharp_files = AugmentedImagesUtil.get_images_file_names_from_folder(folder_sharp, image_exts)

        for image_blurred_file in image_sharp_files:
            assert os.path.isfile(folder_blurred + image_blurred_file), "Blurred file does not exists for the given image..."

        images_files_paths = []
        for image_file in image_sharp_files:
            images_files_paths.append(image_file)
        return images_files_paths
