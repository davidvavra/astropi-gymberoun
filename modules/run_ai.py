from PIL import Image
from os import path
from pycoral.adapters import common
from pycoral.adapters import segment
from pycoral.utils.edgetpu import make_interpreter
import numpy as np
import logging
logger = logging.getLogger("astropi.thread")


class AI():
    """Class containing all stuff around AI model
    """
    def __init__(self, model, folder, base_folder, n_classes=10, colormap=None) -> None:
        """Class initialization

        Args:
            model (str/os.Path): Path to .tflite model file
            n_classes (int, optional): how many classes does the model recognize. Defaults to 10.
            colormap (np.array, optional): colormap mask for recoloring of image mask created by model. Defaults to None -> will use integrated.
            folder (str/os.Path): Folder where to put processed images.
            base_folder (str): Base folder
        """
        # Create EdgeTPU interpreter
        self.interpreter = make_interpreter(model)
        self.interpreter.allocate_tensors()
        # From model get model input parameters (size -> height and width)
        self.width, self.height = common.input_size(self.interpreter) 
        # How many classes do we have
        self.classes = n_classes
        # If no colormap was inputed -> use integrated; else use inputted
        if(colormap == None):
            self.colormap = np.array([[255,106,0],[255,255,0], [7,89,0], [7,255,0], [0,255,215], [90,106,220], [0,0,215], [67,0,219], [227,181,215], [255,106,129]])
        else:
            self.colormap = colormap
        self.folder = folder
        self.base_folder = base_folder
        logger.info(f"Initilized model on EdgeTPU with parameters: [width: {self.width}, height: {self.height}, classes: {self.classes}]")

    def decode_segmentation_mask_internal(self, mask):
        """Function to recolor mask using colormap (internal)

        Args:
            mask (np.array): mask -> model output

        Returns:
            np.array: recolored mask
        """
        # Proces on per color channel basis
        r = np.zeros_like(mask).astype(np.uint8)
        g = np.zeros_like(mask).astype(np.uint8)
        b = np.zeros_like(mask).astype(np.uint8)
        
        # Loop over for each class
        for l in range(0, self.classes):
            idx = mask == l
            r[idx] = self.colormap[l,0]
            g[idx] = self.colormap[l,1]
            b[idx] = self.colormap[l,2]
        # Put color channels back to one image
        self.rgb = np.stack([r,g,b,], axis=2)
        # Return recolored mask
        logger.debug("Decoded segmented image to mask")
        return self.rgb


    def process_image_internal(self, image):
        """Function to process image (internal)

        Args:
            image (str/os.Path): path to image to process

        Returns:
            np.array: np.array of raw model output
        """
        logger.info(f"Processing image {image} on EdgeTPU <fn: process_image_itnernal>")
        img = Image.open(image)
        img = img.resize((self.width,self.height), Image.ANTIALIAS)
        common.set_input(self.interpreter, img)
        logger.debug("Image loaded. Sending image to EdgeTPU")
        self.interpreter.invoke()
        logger.debug("Got result from interpreter.")
        self.result = segment.get_output(self.interpreter)
        self.result = np.argmax(self.result, axis=2)
        return self.result

    def get_raw_image(self):
        """Returns same thing as process_image_interal -> debug only

        Returns:
            np.array: np.array of raw model output
        """
        return self.result

    def get_colored_mask(self):
        """Function that returns colored output from model; primarly internal but can be used externaly

        Returns:
            PIL.Image: colored image mask
        """
        # Load and decode output from model and convert it to PIL.Image
        output_img = Image.fromarray(self.decode_segmentation_mask_internal(self.result.astype(np.uint8)))
        return output_img

    def run_model(self, image):
        """Basically call only this
        Will process image and return only path where final image was saved

        Args:
            image (str/os.Path): path to image to process

        Returns:
            str: where final image is saved
        """
        logger.info(f"Processing image {image} on EdgeTPU <fn: run_model>")
        f_name, f_ext = path.splitext(path.basename(image))
        new_f_name = f_name + "_masked"
        final_f_name = new_f_name + f_ext
        final_path = f"{self.base_folder}/{self.folder}/{final_f_name}"
        self.process_image_internal(image)
        self.get_colored_mask()
        logger.info(f"Successfully processed image {image} on EdgeTPU. Colored image mask will be saved in {final_path}")
        self.output_img.save(self.final_path)
        return final_path
