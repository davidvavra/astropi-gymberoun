import logging
import os

from PIL import Image, ImageDraw

from modules import files
logger = logging.getLogger("astropi")


def process_image(base_folder, counter):
    """Function to crop and preprocess a single image.
    We crop to image to a square and add a circular mask to filter out the window edge.

    Args:
        base_folder: Base folder for all data
        counter: Iteration number

    Returns:
        [] - path to the processed image and boolean whether is usable for classification
    """

    photo_path = f"{base_folder}/{files.LAST_IMAGE_FILE}"
    raw_path = f"{base_folder}/{files.RAW_IMAGES_FOLDER}/image{counter}_raw.jpg"
    try:
        output = _process_image(photo_path)
        if output:  # Image is usable
            logger.debug(f"image number {counter} cropped succesfully")
            processed_image = output[0]
            path = f"{base_folder}/{files.CROPPED_IMAGES_FOLDER}/image_{counter}_croppped.jpg"
            processed_image.save(path)
            return [path, True]
        else: # Image is not usable
            # Save at least raw image
            os.system("cp " + photo_path + " " + raw_path)
            return [raw_path, False]
    except:
        logger.error(f"Error occured while trying to process image number {counter}")
        # Save at least raw image
        os.system("cp " + photo_path + " " + raw_path)
        return [raw_path, False]


def _process_image(input_image, threshold=40, output_size=(1024, 1024), image_mask=False):
    """Function to process single image

    Args:
        input_image (string): Path to image to process
        threshold (int): Threshold for processing. Defaults to 40.
        output_size (tuple, optional): Size of outputed image.
        image_mask (string): Path to image mask (only for cropping). Defaults to False. (=> no mask to crop)

    Returns:
        PIL.Image or bool: PIL.Image if mask successfully created, else False
    """
    im = Image.open(input_image)
    im = im.convert("RGB")
    mask_box = _parent_mask(im, threshold)
    if mask_box != (-1, -1, -1, -1):
        # Usable image/mask
        cropped = im.crop(mask_box)
        cropped = _mask_im(cropped, (5, 5, cropped.size[0] - 5, cropped.size[1] - 5))
        resized = cropped.resize(output_size)
        if image_mask != False:
            mask_image = Image.open(image_mask)
            mask_image = mask_image.crop(mask_box)
            mask_image = _mask_im(mask_image, (5, 5, mask_image.size[0] - 5, mask_image.size[1] - 5))
            mask_resized = mask_image.resize(output_size)
            return [resized, mask_resized]

        return [resized]
    else:
        # Unusable image
        return False


def _make_mask(im, masking_treshold=40):
    """Function to create mask B&W mask from image

    Args:
        im (PIL.Image): Image 
        masking_treshold (int, optional): Threshold for creating mask. Defaults to 40.

    Returns:
        PIL.Image: Mask
    """
    # Convert image to B&W (with shades of gray)
    gray = im.convert("L")
    # If point is brighter than threshold
    mask = gray.point(lambda p: p > masking_treshold and 255)
    return mask

def _parent_mask(img, masking_treshold=40):
    """Function to create square mask for part of image with usable data

    Args:
        img (PIL.Image): Image to work on
        masking_treshold (int, optional): Threshold for creating mask (passed to make_mask()). Defaults to 40.

    Returns:
        tuple: Coordinates of mask box corners (all -1 if none created)
    """
    # Get B&W mask for image
    mask = _make_mask(img, masking_treshold)
    # Calculate rectangular region with most data
    box = mask.getbbox()
    # Crop to wanted data
    cropped = mask.crop(box)
    # Check if image is uable or not
    
    #checking if the image is usable has been removed 
    return box

def _mask_im(im, mask_box):
    """Create a circular region and fill everything else with one uniform color

    Args:
        im (PIL.Image): Image to work with
        mask_box (tuple): Region to crop to
        color (tuple, optional): Color to paint to rest of image. Defaults to (0,0,0).

    Returns:
        PIL.Image: _description_
    """
    # Create a new image with same size as inputed image, will be used as transparency mask
    trans_mask = Image.new("L", im.size, 0)
    # Draw white ellipse in the mask_box region
    draw = ImageDraw.Draw(trans_mask)
    draw.ellipse((mask_box[0] + 5, mask_box[1] + 2) + (mask_box[2] - 5, mask_box[3] - 5), fill=255)
    # Prepare black background
    bac = Image.new("RGB", im.size, (0, 0, 0))
    # Overlay all images with image in the background, black layer in the foreground and ellipse as transparency mask
    return Image.composite(im, bac, trans_mask)
