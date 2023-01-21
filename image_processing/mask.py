import time
import numpy
import os


from PIL import Image, ImageDraw


def make_mask(im, masking_treshold = 40):
    gray = im.convert("L")
    mask = gray.point(lambda p: p > masking_treshold  and 255)
    return mask



def mask_im(im, mask_box, color = (0,0,0)):
    test = Image.new("L", im.size, 0)
    draw = ImageDraw.Draw(test)
    draw.ellipse((mask_box[0] + 5, mask_box[1] + 2) + (mask_box[2] - 5, mask_box[3] - 5), fill=255)
    bac = Image.new("RGB", im.size, (0,0,0))
    return Image.composite(im, bac, test)



def is_good(cropped_mask, min_color = 40):
    if cropped_mask.size[0]/cropped_mask.size[1] > 1.05 or cropped_mask.size[0]/cropped_mask.size[1] < 0.95:
        return False
    arr = numpy.array(cropped_mask)
    if numpy.mean(arr) < min_color:
        return False 
    return True




def parent_mask(img, masking_treshold):
    mask = make_mask(img, masking_treshold)
    box = mask.getbbox()
    cropped = mask.crop(box)
    if is_good(cropped):
        return box
    else:
        return "sucken dicken"






def find_mask(input_folder, treshold, tolerance = 30):
    masks = []
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg"):
            im = Image.open(input_folder + "/" + filename)
            im = im.convert("RGB")
            mask_box = parent_mask(im, treshold)
            if mask_box != "sucken dicken":
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




    