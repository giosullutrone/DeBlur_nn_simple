if __name__ == "__main__":
    import argparse
    from src.DeBlurSingleNet import DeBlurSingleNet
    from src.DeBlurAdversarialNet import DeBlurAdversarialNet
    from src.Generator import Generator
    from src.GeneratorAdversarial import GeneratorForGAN, GeneratorForDiscriminator
    from src.Util import get_fixed_path
    import tensorflow.keras.backend as K
    import os

    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--datasets_folder", help="Location of the datasets folders", type=str, required=True)
    parser.add_argument("-o", "--save_folder", help="Where to save the models", type=str, required=True)

    parser.add_argument("-l", "--load_path", help="If and from where to load the model", type=str,
                        required=False, default=None)

    parser.add_argument("-e", "--epochs", help="Epochs to train the model for", type=int, required=True)
    parser.add_argument("-bs", "--batch_size", help="Batch size", type=int, required=True)

    parser.add_argument("-mt", "--model_type", help="Model type to use. Default: \"s\"", type=str, required=False, default="s")

    parser.add_argument("-s", "--size", help="Input size of each patch. Default: 224 224", nargs=2, type=int, required=False, default=(224, 224))

    args = parser.parse_args()

    datasets_folder = get_fixed_path(args.datasets_folder, replace_backslash=True, add_backslash=True)
    save_folder = get_fixed_path(args.save_folder, replace_backslash=True, add_backslash=True)

    assert args.epochs > 0, "Epochs are less than 0..."

    ####################################################################################################################
    # DeBlur Single network / DeBlur Adversarial Net
    ####################################################################################################################
    K.clear_session()

    if args.model_type != "a":
        print("ok")
        net = DeBlurSingleNet(input_shape=(args.size[0], args.size[1], 3), model_type=args.model_type)

        os.makedirs(save_folder, exist_ok=True)

        net.train_model(epochs=args.epochs,
                        folder_weights_save=save_folder,
                        path_weights_load=args.load_path,
                        generator=Generator(
                            folder_sharp_images=datasets_folder + "Training/" + "Sharp/",
                            folder_blurred_images=datasets_folder + "Training/" + "Blurred/",
                            batch_size=args.batch_size,
                            image_exts=(".jpg", ".png", ".jpeg"),
                            shuffle=True
                        ),
                        generator_validation=Generator(
                            folder_sharp_images=datasets_folder + "Validation/" + "Sharp/",
                            folder_blurred_images=datasets_folder + "Validation/" + "Blurred/",
                            batch_size=args.batch_size,
                            image_exts=(".jpg", ".png", ".jpeg"),
                            shuffle=True
                        ))
    else:
        net = DeBlurAdversarialNet(input_shape=(args.size[0], args.size[1], 3))

        os.makedirs(save_folder, exist_ok=True)

        net.train_model(epochs=args.epochs,
                        folder_weights_save=save_folder,
                        path_weights_load=args.load_path,
                        generator_discriminator=GeneratorForDiscriminator(
                            folder_sharp_images=datasets_folder + "Training/" + "Sharp/",
                            folder_blurred_images=datasets_folder + "Training/" + "Blurred/",
                            batch_size=args.batch_size,
                            image_exts=(".jpg", ".png", ".jpeg"),
                            shuffle=True
                        ),
                        generator_gan=GeneratorForGAN(
                            folder_sharp_images=datasets_folder + "Training/" + "Sharp/",
                            folder_blurred_images=datasets_folder + "Training/" + "Blurred/",
                            batch_size=args.batch_size,
                            image_exts=(".jpg", ".png", ".jpeg"),
                            shuffle=True
                        ),
                        generator_validation=GeneratorForGAN(
                            folder_sharp_images=datasets_folder + "Validation/" + "Sharp/",
                            folder_blurred_images=datasets_folder + "Validation/" + "Blurred/",
                            batch_size=args.batch_size,
                            image_exts=(".jpg", ".png", ".jpeg"),
                            shuffle=True
                        ))
