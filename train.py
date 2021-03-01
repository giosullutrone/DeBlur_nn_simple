if __name__ == "__main__":
    import argparse
    from src.DeBlurNetwork import DeBlurNetwork
    from src.Util import get_fixed_path
    import tensorflow.keras.backend as K
    import src.Nets
    import os

    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--datasets_folder", help="Location of the \"Datasets\" folder", type=str, required=True)
    parser.add_argument("-o", "--save_folder", help="Where to save the models", type=str, required=True)

    parser.add_argument("-l", "--load_folder", help="If and from where to load the models' weights", type=str,
                        required=False, default=None)

    parser.add_argument("-e", "--epochs", help="Epochs for the pre tracking network", type=int,
                        required=False, default=20)

    parser.add_argument("-spe", "--steps_per_epoch", help="Steps per epoch", type=int, required=False, default=128)
    parser.add_argument("-bs", "--batch_size", help="Batch size", type=int, required=False, default=2)

    parser.add_argument("-dc", "--do_checkpoint", help="Should save checkpoints", type=bool, required=False, default=True)

    parser.add_argument("-s", "--size", help="Input size of each patch", nargs=2, type=int, required=False, default=(224, 224))

    args = parser.parse_args()

    datasets_folder = get_fixed_path(args.datasets_folder, replace_backslash=True, add_backslash=True)
    save_folder = get_fixed_path(args.save_folder, replace_backslash=True, add_backslash=True)
    load_folder = get_fixed_path(args.load_folder, replace_backslash=True, add_backslash=True) if args.load_folder is not None else None

    ####################################################################################################################
    # DeBlur network
    ####################################################################################################################
    if args.epochs > 0:
        K.clear_session()

        net = DeBlurNetwork(src.Nets.generate_model(input_shape=(args.size[0], args.size[1], 3), L2=0.0))

        net.get_model().summary()

        os.makedirs(save_folder, exist_ok=True)

        net.train_model(epochs=args.epochs,
                        steps_per_epoch=args.steps_per_epoch,
                        folder_weights_save=save_folder,
                        folder_weights_load=load_folder,
                        do_checkpoint=args.do_checkpoint,
                        generator=DeBlurNetwork.generator(
                            folder_sharp_images=datasets_folder + "Training/" + "Sharp/",
                            folder_blurred_images=datasets_folder + "Training/" + "Blurred/",
                            batch_size=args.batch_size,
                            image_ext=(".jpg", ".png", ".jpeg"),
                            section_size=(args.size[0], args.size[1], 3)
                        ),
                        generator_validation=DeBlurNetwork.generator(
                            folder_sharp_images=datasets_folder + "Validation/" + "Sharp/",
                            folder_blurred_images=datasets_folder + "Validation/" + "Blurred/",
                            batch_size=args.batch_size,
                            image_ext=(".jpg", ".png", ".jpeg"),
                            section_size=(args.size[0], args.size[1], 3)
                        ))
