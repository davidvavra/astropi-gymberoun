# Import libraries
import os

import numpy
from PIL import Image, ImageDraw


def make_mask(im, masking_treshold = 40):
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

def is_good(cropped_mask, orig_size = (1296, 972), min_color = 100):
    """Check if image is usable or not

    Args:
        cropped_mask (PIL.Image): Image cropped to wanted part
        min_color (int, optional): Minimum wanted average of pixels in region. Defaults to 40.

    Returns:
        bool: True : image is usable, otherwise False
    """
    # Check if image is +/- square
    if cropped_mask.size[0]/cropped_mask.size[1] > 1.4 or cropped_mask.size[0]/cropped_mask.size[1] < 0.7:
        # Image is not good if not square
        return False
    # Create numpy array from image


    arr = numpy.array(cropped_mask)
    # If average value in image is less than min_color
    if numpy.mean(arr) < min_color:
        # Not good
        return False 

    # check if the mask is at least 55 by 70 percent of the original image
    if cropped_mask.size[0] < 0.55*orig_size[0] or cropped_mask.size[1] < 0.7*orig_size[1]:
        return False
        
    # Else image is good
    return True

def parent_mask(img, masking_treshold = 40):
    """Function to create square mask for part of image with usable data

    Args:
        img (PIL.Image): Image to work on
        masking_treshold (int, optional): Threshold for creating mask (passed to make_mask()). Defaults to 40.

    Returns:
        tuple: Coordinates of mask box corners (all -1 if none created)
    """
    # Get B&W mask for image
    mask = make_mask(img, masking_treshold)
    # Calculate rectangular region with most data
    box = mask.getbbox()
    # Crop to wanted data
    cropped = mask.crop(box)
    # Check if image is uable or not
    if is_good(cropped):
        # Return coordinate of box corners
        return box
    else:
        # Not usable => return -1s
        return (-1,-1,-1,-1)

def mask_im(im, mask_box, color = (0,0,0)):
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
    bac = Image.new("RGB", im.size, (0,0,0))
    # Overlay all images with image in the background, black layer in the foreground and ellipse as transparency mask
    return Image.composite(im, bac, trans_mask)

def m_process_image(input_image, threshold = 40, output_size = (800,800), image_mask = False):
    """Function to process single image

    Args:
        input_image (string): Path to image to process
        threshold (int): Threshold for processing. Defaults to 40.
        output_size (tuple, optional): Size of outputed image. Defaults to (800,800).
        image_mask (string): Path to image mask (only for cropping). Defaults to False. (=> no mask to crop)

    Returns:
        PIL.Image or bool: PIL.Image if mask successfully created, else False
    """
    im = Image.open(input_image)
    im = im.convert("RGB")
    mask_box = parent_mask(im, threshold)
    if mask_box != (-1,-1,-1,-1):
        # Usable image/mask
        cropped = im.crop(mask_box)
        cropped = mask_im(cropped, (5, 5, cropped.size[0]-5, cropped.size[1]-5))
        resized = cropped.resize(output_size)
        if image_mask != False:
            mask_image = Image.open(image_mask)
            mask_image = mask_image.crop(mask_box)
            mask_image = mask_im(mask_image, (5, 5, mask_image.size[0]-5, mask_image.size[1]-5))
            mask_resized = mask_image.resize(output_size)
            return [resized, mask_resized]

        return [resized]
    else:
        # Unusable image
        return False

def find_mask(input_folder, treshold, tolerance = 30):
    masks = []
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg"):
            im = Image.open(input_folder + "/" + filename)
            im = im.convert("RGB")
            mask_box = parent_mask(im, treshold)
            if mask_box != (-1,-1,-1,-1):
                masks.append(mask_box)
        if len(masks) > 2:
            new = []
            for i in range(4):
                print(i)
                new.append (int((masks[0][i] + masks[1][i] + masks[2][i]) / 3))

            passed = True
            for i in range(3):
                for j in range(4):
                    if masks[i][j] > new[j] + tolerance or masks[i][j] < new[j] - tolerance:
                        masks = []
                        passed = False
            if passed:
                return new

def process_image(photo, base_folder, cropped_folder, raw_folder, counter, logger):
    """Function to crop and preprocess a single image during the experiment

    Args:
        photo (string) - relative path to image to be processed 
        base_folder (string) - relative path to base folder
        cropped_folder (string) - relative path to the output folder for cropped_imgaes
        raw_folder (string) - relative path to the folder to save unprocessable images in
        counter (int) - current count of the images
        logger - logging object

    Returns:
        string - path to the processed image or None if the image was deemed unusable;
    """
    try:
        outputimage = m_process_image(os.path.join(base_folder, photo))
        if outputimage: # if the image is unusable, m_process_image returns False
            logger.debug(f"image number {counter} cropped succesfully")
            outputimage = outputimage[0] # m_process image returns a list
            filename2 = os.path.join(cropped_folder, f"image_{counter}_croppped.jpg")
            outputimage.save(filename2)

            return filename2
        # if the image is considered unusable we do nothing
        else:
            logger.debug(f"image number {counter} could not be cropped")
            photo.save(os.path.join(raw_folder, f"image{counter}_raw.jpg"))
            return None
    except:
        logger.error(f"Error occured while trying to process image number {counter}")
        photo.save(os.path.join(raw_folder, f"image{counter}_raw.jpg"))
        return None

if __name__ == "__main__":
    output_size = (800, 800)

    output_folder = "out"
    input_folder = os.getcwd()
    
    os.mkdir(output_folder)
    masking_treshold = 40

    mask_box = find_mask(input_folder, masking_treshold)

    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg"):
            im = Image.open(input_folder + "/" + filename)
            im = im.convert("RGB")
            check_box = parent_mask(im, masking_treshold)
            
            
            cropped = im.crop(mask_box)
            cropped = mask_im(cropped, (5, 5, cropped.size[0]-5, cropped.size[1]-5))
            cropped = cropped.resize(output_size)
            cropped.save(output_folder + "/" + filename)
            
