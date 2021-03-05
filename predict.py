if __name__ == "__main__":
    import argparse
    from src.AugmentedImage import AugmentedImage
    from src.AugmentedImagesUtil import AugmentedImagesUtil
    from src.Util import get_fixed_path
    import tensorflow.keras.backend as K
    from src.DeBlurNetwork import DeBlurNetwork

    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input_folder", help="Location of the \"Input\" folder", type=str, required=True)
    parser.add_argument("-o", "--output_folder", help="Location of the \"Output\" folder", type=str, required=True)
    parser.add_argument("-l", "--load_path", help="From where to load the models' weights", type=str, required=True)

    parser.add_argument("-s", "--size", help="Input size of each patch", nargs=2, type=int, required=False, default=(224, 224))

    args = parser.parse_args()

    input_folder = get_fixed_path(args.input_folder, replace_backslash=True, add_backslash=True)
    output_folder = get_fixed_path(args.output_folder, replace_backslash=True, add_backslash=True)

    ####################################################################################################################
    # DeBlur network
    ####################################################################################################################
    K.clear_session()
    net = DeBlurNetwork(input_shape=(args.size[0], args.size[1], 3))
    net.load_weights(args.load_path)

    image_files = AugmentedImagesUtil.get_images_file_names_from_folder(input_folder, image_exts=(".jpg", ".png", ".jpeg"))

    for image_file in image_files:
        pred = net.predict_from_image(AugmentedImage.image_from_file(input_folder + image_file), section_size=tuple(args.size))
        AugmentedImage.image_to_file(pred, image_path=output_folder + image_file)
