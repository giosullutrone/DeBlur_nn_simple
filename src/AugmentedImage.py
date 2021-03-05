class AugmentedImage:
    """Class that lets you manipulate an image"""

    @staticmethod
    def get_image_width(image):
        return image.shape[1]

    @staticmethod
    def get_image_height(image):
        return image.shape[0]

    @staticmethod
    def image_from_file(image_path, grayscale=False):
        """Get image from file path"""
        import cv2
        if grayscale:
            image = cv2.imread(image_path, 0)
        else:
            image = cv2.imread(image_path)
        return image

    @staticmethod
    def get_section(image, x, y, section_size):
        """Returns a numpy array of a section of the image with the size specified by section_size starting from (x, y)"""
        image_width = AugmentedImage.get_image_width(image)
        image_height = AugmentedImage.get_image_height(image)
        section_width = section_size[0]
        section_height = section_size[1]

        assert x + section_width <= image_width and y + section_height <= image_height, "Section exceeds image dimensions..."

        return image[int(y):int(y+section_height), int(x):int(x+section_width)]

    @staticmethod
    def get_sections_annotated(image, section_size):
        """Returns a tuple containing all possible sections of the image, as (x, y, np array)"""
        image_width = AugmentedImage.get_image_width(image)
        image_height = AugmentedImage.get_image_height(image)
        section_width = section_size[0]
        section_height = section_size[1]

        assert image_width >= section_width and image_height >= section_height, "Section size bigger than image..."

        number_of_sections_x = image_width / section_width
        number_of_sections_y = image_height / section_height

        sections = []

        for i in range(int(number_of_sections_x)):
            for j in range(int(number_of_sections_y)):
                x = int(i*section_width)
                y = int(j*section_height)
                sections.append((x, y, AugmentedImage.get_section(image, x, y, section_size)))

        #Get overlapping sections if needed
        if number_of_sections_x - int(number_of_sections_x) > 0:
            for j in range(int(number_of_sections_y)):
                x = image_width - section_width
                y = int(j*section_height)

                sections.append((x, y, AugmentedImage.get_section(image, x, y, section_size)))

        if number_of_sections_y - int(number_of_sections_y) > 0:
            for i in range(int(number_of_sections_x)):
                x = int(i*section_width)
                y = image_height - section_height

                sections.append((x, y, AugmentedImage.get_section(image, x, y, section_size)))

        if number_of_sections_x - int(number_of_sections_x) > 0 and number_of_sections_y - int(number_of_sections_y) > 0:
            x = image_width - section_width
            y = image_height - section_height

            sections.append((x, y, AugmentedImage.get_section(image, x, y, section_size)))
        return sections

    @staticmethod
    def image_from_sections_annotated(base_image, sections):
        """Overwrite image with the pixels from annotated sections"""
        section_width = AugmentedImage.get_image_width(sections[0][2])
        section_height = AugmentedImage.get_image_height(sections[0][2])

        for section in sections:
            x, y, section_image = section
            base_image[y:y+section_height, x:x+section_width] = section_image.astype(base_image.dtype)
        return base_image

    @staticmethod
    def image_to_file(image, image_path, image_size=None, grayscale=False, invert=False):
        import cv2

        if image_size is not None:
            image = cv2.resize(image, image_size, interpolation=cv2.INTER_AREA)
        try:
            if grayscale:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            pass
        if invert:
            image = cv2.bitwise_not(image)
        cv2.imwrite(image_path, image)
