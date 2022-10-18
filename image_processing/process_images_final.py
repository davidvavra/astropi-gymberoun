# import knihoven
import cv2
import numpy as np 
import os
import sys
from PIL import Image
import time
import shutil

# Složka se soubory ke zpracování
#directory = "pre_processed0"
directory = "inImages"


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
    bottom = (r.astype(float) + b.astype(float))
    bottom[bottom==0] = 0.01
    ndvi = (b.astype(float) - r) / bottom
    return ndvi

# Funkční funkce která zpracuje obrázek na NDVI
def process_img(img_path, index):
    # Jméno výstupního souboru
    name = "img" + str(index)
    print("Processing image: " + img_path + " *-*-* Outputed as: " + "output/" + ("img" + str(index)) + "/" + name + " + _ndvi | _c-mapped | _original.png")
    # Vytvoření složky pro výstup
    os.mkdir("output/img" + str(index))

    # Načtení obrázku
    original = cv2.imread(img_path)
    #display(original, 'Original')
    # Zvýšení kontrastu
    contrasted = contrast_stretch(original)
    #display(contrasted, 'Contrasted original')
    #cv2.imwrite('contrasted.png', contrasted)
    # Počítání NDVI
    ndvi = calc_ndvi(contrasted)
    #display(ndvi, 'NDVI')
    # Kontrast NDVI
    ndvi_contrasted = contrast_stretch(ndvi)
    #display(ndvi_contrasted, 'NDVI Contrasted')
    # Color mapping NDVI obrázku pro snazší lidské zpracování
    color_mapped_prep = ndvi_contrasted.astype(np.uint8)
    color_mapped_image = cv2.applyColorMap(color_mapped_prep, cv2.COLORMAP_JET)
    #display(color_mapped_image, 'Color mapped')
    # Zapsání výstupních souborů
    cv2.imwrite("output/" + name + "/" + name + '_ndvi.png', ndvi_contrasted)
    cv2.imwrite("output/" + name + "/" + name + '_c-mapped.png', color_mapped_image)
    cv2.imwrite("output/" + name + "/" + name + '_original.png', original)
    # Vytvoření ořízlých variant
    crop("output/" + name + "/" + name + "_ndvi.png")
    crop("output/" + name + "/" + name + "_c-mapped.png")
    crop("output/" + name + "/" + name + "_original.png")

# Funke která ořízne obrázek a uloží ho jako kopii
def crop(file):
    # Načtení masky
    mask = cv2.imread("mask.jpg", cv2.IMREAD_UNCHANGED)
    mask = np.uint8(mask)
    # Načtení obrázku
    im = cv2.imread(file)
    im = np.uint8(im)

    # Oříznutí pomocí masky
    cropped = cv2.bitwise_and(im, mask)
    # Uložení
    cropped = Image.fromarray(cropped.astype(np.uint8))
    cropped.save(str(file).split(".")[0] + "-cropped.png")

# Při řazení souborů... pokud soubory "soubor (1).jpg", "soubor (2).jpg"
def sortKey(val):
    ret = int(val[val.index("(")+1:val.index(")")])
    return ret

# Průměr hodnot v poli
def Average(lst):
    return sum(lst) / len(lst)

# Příprava místa pro výstup
shutil.rmtree("output")
os.mkdir("output")

# Walk thru entire folder
files = os.listdir(directory)
files.sort()
# Log times to process each image
times = []
# Projití celé složky a zpracování každého obrázku
for x,file_name in enumerate(files):
    start_time = time.time()
    process_img(directory + "/" + file_name, x)
    print("Procesing this file took: " + str(time.time() - start_time) + "s")
    times.append(time.time() - start_time)

# Average processing time per image
print("-----------\nAverage time per image: " + str(Average(times)))