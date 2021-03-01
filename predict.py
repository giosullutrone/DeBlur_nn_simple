if __name__ == "__main__":
    import argparse
    import os
    from src.AugmentedImage import AugmentedImage
    from src.Util import get_fixed_path
    import tensorflow.keras.backend as K
    import src.Nets
    from src.DeBlurNetwork import DeBlurNetwork

    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input_folder", help="Location of the \"Input\" folder", type=str, required=True)
    parser.add_argument("-o", "--output_folder", help="Location of the \"Output\" folder", type=str, required=True)
    parser.add_argument("-m", "--models_folder", help="From where to load the models' weights", type=str, required=True)

    parser.add_argument("-s", "--size", help="Input size of each patch", nargs=2, type=int, required=False,
                        default=(224, 224))

    args = parser.parse_args()

    input_folder = get_fixed_path(args.input_folder, replace_backslash=True, add_backslash=True)
    output_folder = get_fixed_path(args.output_folder, replace_backslash=True, add_backslash=True)
    models_folder = get_fixed_path(args.models_folder, replace_backslash=True, add_backslash=True)

    ####################################################################################################################
    # Tracking network
    ####################################################################################################################
    K.clear_session()

    net = DeBlurNetwork(src.Nets.generate_model(input_shape=(args.size[0], args.size[1], 3), L2=0.0))

    net.load_weights(models_folder)

    ####################################################################################################################
    # Prediction
    ####################################################################################################################
    pred = net.predict_from_image_full(AugmentedImage.image_from_file("H:\Kaggle/0.jpeg"), section_size=tuple(args.size))
    AugmentedImage.image_to_file(pred, image_path="H:\BlurDataset\Test/0.jpg")
    pred = net.predict_from_image(AugmentedImage.image_from_file("H:\BlurDataset\Validation\Blurred/0.jpg"))
    AugmentedImage.image_to_file(pred, image_path="H:\BlurDataset\Test/1.jpg")
    pred = net.predict_from_image(AugmentedImage.image_from_file("H:\BlurDataset\Training\Blurred/10.jpg"))
    AugmentedImage.image_to_file(pred, image_path="H:\BlurDataset\Test/10.jpg")

    import cv2
    #pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
    #AugmentedImage.image_to_file(pred, image_path="H:\BlurDataset\Test/0.jpg")
