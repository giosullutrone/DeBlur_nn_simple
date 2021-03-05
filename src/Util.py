def get_fixed_path(path, replace_backslash=False, add_backslash=False):
    if replace_backslash:
        path = path.replace("\\", "/")

    if add_backslash:
        if not path.endswith("/"):
            path += "/"
    return path


"""
def predict_from_image_full(self, image, section_size=(416, 416)):
    Returns the reconstructed image from predictions
    from src.AugmentedImage import AugmentedImage
    import numpy as np

    aug_image = AugmentedImage(image)

    sections = aug_image.get_sections_annotated(section_size)

    to_predict = []
    for section in sections:
        _, _, image_section = section
        to_predict.append(image_section)

    predictions = self.__model.predict(DeBlurNetwork.__pre_process(np.array(to_predict)))

    sections_predicted = []
    for section, prediction in zip(sections, predictions):
        x, y, _ = section
        image_predicted = prediction
        sections_predicted.append((x, y, DeBlurNetwork.__post_process(image_predicted)))

    return aug_image.image_from_sections_annotated(sections_predicted)
"""
