# import knihoven
import cv2
import numpy as np 
import notify2
import os
import sys
import argparse
from PIL import Image
import time
import shutil

class text:
    '''Colors class:reset all colors with colors.reset; two
    sub classes fg for foreground
    and bg for background; use as colors.subclass.colorname.
    i.e. colors.fg.red or colors.bg.greenalso, the generic bold, disable,
    underline, reverse, strike through,
    and invisible work with the main class i.e. colors.bold'''
    reset = '\033[0m'
    bold = '\033[01m'
    disable = '\033[02m'
    underline = '\033[04m'
    reverse = '\033[07m'
    strikethrough = '\033[09m'
    invisible = '\033[08m'
 
    class fg:
        black = '\033[30m'
        red = '\033[31m'
        green = '\033[32m'
        orange = '\033[33m'
        blue = '\033[34m'
        purple = '\033[35m'
        cyan = '\033[36m'
        lightgrey = '\033[37m'
        darkgrey = '\033[90m'
        lightred = '\033[91m'
        lightgreen = '\033[92m'
        yellow = '\033[93m'
        lightblue = '\033[94m'
        pink = '\033[95m'
        lightcyan = '\033[96m'
 
    class bg:
        black = '\033[40m'
        red = '\033[41m'
        green = '\033[42m'
        orange = '\033[43m'
        blue = '\033[44m'
        purple = '\033[45m'
        cyan = '\033[46m'
        lightgrey = '\033[47m'

# Zobrazení obrázku
def display(image, image_name):
    image = np.array(image, dtype=float)/float(255)
    shape = image.shape
    height = int(shape[0] / 2)
    width = int(shape[1] / 2)
    image = cv2.resize(image, (width, height))
    cv2.namedWindow(image_name)
    cv2.imshow(image_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# idk something stolen
def contrast_stretch(im):
    in_min = np.percentile(im, 5)
    in_max = np.percentile(im, 95)

    out_min = 0.0
    out_max = 255.0

    out = im - in_min
    out *= ((out_min - out_max) / (in_min - in_max))
    out += in_min

    return out

# another stolen code, calcualtes NDVI
def calc_ndvi(image):
    b, g, r = cv2.split(image)
    bottom = (r.astype(float) + b.astype(float)) # Red + NIR
    bottom[bottom==0] = 0.01
    ndvi = (b.astype(float) - r) / bottom
    return ndvi

# Funkční funkce která zpracuje obrázek na NDVI
def process_img(img_path, index, single):
    # Jméno výstupního souboru
    name = "img" + str(index)
    #print(f"{text.fg.blue}Processing image: " + img_path + " *-*-* Outputed as: " + "output/" + ("img" + str(index)) + "/" + name + f" + _ndvi | _c-mapped | _original.png{text.reset}")

    # Načtení obrázku
    original = cv2.imread(img_path)
    #display(original, 'Original')
    # Zvýšení kontrastu
    contrasted = contrast_stretch(original)
    #display(contrasted, 'Contrasted original')
    #cv2.imwrite('contrasted.png', contrasted)

    # Výpočet NDVI
    ndvi = calc_ndvi(contrasted)
    # Kontrast NDVI
    ndvi_contrasted = contrast_stretch(ndvi)
    # Color mapping NDVI obrázku pro snazší lidské zpracování
    color_mapped_prep = ndvi_contrasted.astype(np.uint8)
    color_mapped_image_ndvi = cv2.applyColorMap(color_mapped_prep, cv2.COLORMAP_JET)



    # Pokud je soubor jediný 
    if(single):
        name = os.path.basename(img_path).split('/')[-1].split('.')[:-1][0]
        # Zapsání výstupních souborů
        cv2.imwrite("output/" + name + '_ndvi.png', ndvi_contrasted)
        cv2.imwrite("output/" + name + '_c-mapped.png', color_mapped_image_ndvi)
        cv2.imwrite("output/" + name + '_original.png', original)
        # Vytvoření ořízlých variant
        crop("output/" + name + "_ndvi-cropped.png", "output/" + name + '_ndvi.png')
        crop("output/" + name + "_c-mapped-cropped.png", "output/" + name + '_c-mapped.png')
        crop("output/" + name + "_original-cropped.png", "output/" + name + '_original.png')    
    # Soubor není jediný, zpracovávání složky
    else:
        # Vytvoření složky pro výstup
        os.mkdir("output/img" + str(index))
        # Zapsání výstupních souborů
        cv2.imwrite("output/" + name + "/" + name + '_ndvi.png', ndvi_contrasted)
        cv2.imwrite("output/" + name + "/" + name + '_c-mapped.png', color_mapped_image_ndvi)
        cv2.imwrite("output/" + name + "/" + name + '_original.png', original)
        # Vytvoření ořízlých variant
        crop("output/" + name + "/" + name + "_ndvi-cropped.png", "output/" + name + "/" + name + '_ndvi.png')
        crop("output/" + name + "/" + name + "_c-mapped-cropped.png", "output/" + name + "/" + name + '_c-mapped.png')
        crop("output/" + name + "/" + name + "_original-cropped.png", "output/" + name + "/" + name + '_original.png')

# Funkce která ořízne obrázek a uloží ho jako kopii
def crop(file, input_image):
    # Načtení masky
    mask = Image.open("mask.png")
    # Načtení obrázku
    img = Image.open(input_image)

    # Oříznutí pomocí masky
    img.paste(mask, (0,0), mask=mask)
    # Uložení
    img.save(file)
    

# Při řazení souborů... pokud soubory "soubor (1).jpg", "soubor (2).jpg"
def sortKey(val):
    ret = int(val[val.index("(")+1:val.index(")")])
    return ret

# Průměr hodnot v poli
def Average(lst):
    return sum(lst) / len(lst)

def processFolder():
    main_time = time.time()
    print(f'{text.bold}{text.fg.yellow}Processing all images in {text.underline}{directory}{text.reset}{text.bold}{text.fg.yellow} folder{text.reset}\n')
    # Kontrola jestli složka pro výstup exituje
    if(os.path.isdir("output")):
        # Existuje
        # Příprava místa pro výstup
        shutil.rmtree("output")
        os.mkdir("output")
    else:
        # Neexistuje - vytvořit
        os.mkdir("output")

    # Walk thru entire folder
    files = os.listdir(directory)
    total_files = len(files)
    files.sort(key=sortKey)
    # Log times to process each image
    times = []
    # Projití celé složky a zpracování každého obrázku
    for x,file_name in enumerate(files):
        start_time = time.time()
        if(x % 2 == 0):
            img_path = directory + "/" + file_name
            name = "img" + str(x)
            print(f"{text.fg.blue}Processing image: " + img_path + " *-*-* Outputed as: " + "output/" + ("img" + str(x)) + "/" + name + f" + _ndvi | _c-mapped | _original.png")
            process_img(directory + "/" + file_name, x, False)
            print("Procesing this file took: " + str(time.time() - start_time) + "s" + text.reset)
        else:
            img_path = directory + "/" + file_name
            name = "img" + str(x)
            print(f"{text.fg.cyan}Processing image: " + img_path + " *-*-* Outputed as: " + "output/" + ("img" + str(x)) + "/" + name + f" + _ndvi | _c-mapped | _original.png")
            process_img(directory + "/" + file_name, x, False)
            print("Procesing this file took: " + str(time.time() - start_time) + "s" + text.reset)
        times.append(time.time() - start_time)

    n = notify2.Notification("AstroPi Dataset Creator", message="All images have been processed!")
    n.set_timeout(5000)
    n.show()
    # Average processing time per image
    print(text.fg.green + f"\n-----------\nProcessed {total_files} images\nTotal time {time.time()-main_time}s\nAverage time per image: " + str(Average(times)) + "s" + text.reset)

# Pokud je program spuštěn
if(__name__ == "__main__"):
    # Arguments parser 
    parser = argparse.ArgumentParser(description=f"{text.bold}Program for processing images from AstroPi for ML by calculating NDVI and cropping them.{text.reset}", epilog=f"{text.bold}{text.underline}Blboun3{text.reset}{text.underline} - AstroPi 2022/23: Barrande{text.reset}")
    parser.add_argument("-f", "--file", help = "input single file")
    parser.add_argument("-F", "--Folder", help = "input all image files in folder. If -f also present -F option will be prioritized.")
    args = parser.parse_args()

    # Notifier
    notify2.init("AstroPi Dataset Creator")

    # Pokud byla zadána složka
    if(args.Folder):
        args.file = None
        directory = args.Folder
        n = notify2.Notification("AstroPi Dataset Creator", message="Image processing has started!")
        n.set_timeout(5000)
        n.show()
        processFolder()
    elif(args.file):
        file = args.file
        print(f"{text.fg.yellow}{text.bold}Processing image {text.underline}{file}{text.reset}")
        n = notify2.Notification("AstroPi Dataset Creator", message=f"Processing of image {file} has started!")
        n.set_timeout(5000)
        n.show()
        args.Folder = None
        master_time = time.time()
        # Kontrola jestli složka pro výstup exituje
        if(os.path.isdir("output")):
            # Existuje
            # Příprava místa pro výstup
            shutil.rmtree("output")
            os.mkdir("output")
        else:
            # Neexistuje - vytvořit
            os.mkdir("output")
        process_img(file, 0, True)
        n = notify2.Notification("AstroPi Dataset Creator", message="Image has been processed!")
        n.set_timeout(5000)
        n.show()
        print(f"{text.fg.green}-----------\nImage {text.bold}{file}{text.reset}{text.fg.green} processed successfuly\nTotal time: {time.time() - master_time}s{text.reset}")


