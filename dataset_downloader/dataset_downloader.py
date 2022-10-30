import base64
from concurrent.futures import ThreadPoolExecutor
from os import mkdir, rmdir, listdir
import shutil
import cv2
import logging
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as ec
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException
from time import perf_counter
from PIL import Image
from io import BytesIO
import logging.config
import notify2
from GPSPhoto import gpsphoto


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

def get_url(latitude, longtitude, zoom=8):
    """
    Function to generate URL for sentinel playground based of latitude and longtitude

    Image [R: 04;G: 08;B: 08]

    Args:
        latitude (int): latitude
        longtitude (int): longtitude
        zoom (int, optional): Zoom level. Defaults to 8.

    Returns:
        string: Generated URL
    """
    return f'https://apps.sentinel-hub.com/sentinel-playground/?source=S2&lat={latitude}&lng={longtitude}&zoom={zoom}&preset=CUSTOM&layers=B04,B08,B08&maxcc=20&gain=1.0&gamma=1.0&time=2022-04-01%7C2022-10-29&atmFilter=&showDates=false&evalscript=cmV0dXJuIFtCMDQqMi41LEIwOCoyLjUsQjA4KjIuNV0%3D'

def accept_tos(driver):
    """
    Accepts Sentinel playground Terms of Service
    """
    driver.find_element(By.CLASS_NAME, 'accept-btn').click()


def generate_image(driver):
    """
    Finds and clicks button for generating image on sentinel playground site
    """
    driver.find_element(By.CLASS_NAME, 'fa-print').click()

def get_image_url(driver):
    """
    Function to get generated image URL 

    Returns:
        string: Image URL
    """
    url = driver.find_element(By.ID, 'sentinelImage').get_attribute('src')
    return url

def get_file_content_chrome(driver, uri):
    """
    Function to download blob image from chrome

    Args:
        driver (selenium webdriver): browser instance to use
        uri (string): Blob URL of the image

    Raises:
        Exception: Request failed with status %s

    Returns:
        bytes: Decoded bytes of image 
    """
    result = driver.execute_async_script('''
        var uri = arguments[0];
        var callback = arguments[1];
        var toBase64 = function(buffer){for(var r,n=new Uint8Array(buffer),t=n.length,a=new Uint8Array(4*Math.ceil(t/3)),i=new Uint8Array(64),o=0,c=0;64>c;++c)i[c]="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/".charCodeAt(c);for(c=0;t-t%3>c;c+=3,o+=4)r=n[c]<<16|n[c+1]<<8|n[c+2],a[o]=i[r>>18],a[o+1]=i[r>>12&63],a[o+2]=i[r>>6&63],a[o+3]=i[63&r];return t%3===1?(r=n[t-1],a[o]=i[r>>2],a[o+1]=i[r<<4&63],a[o+2]=61,a[o+3]=61):t%3===2&&(r=(n[t-2]<<8)+n[t-1],a[o]=i[r>>10],a[o+1]=i[r>>4&63],a[o+2]=i[r<<2&63],a[o+3]=61),new TextDecoder("ascii").decode(a)};
        var xhr = new XMLHttpRequest();
        xhr.responseType = 'arraybuffer';
        xhr.onload = function(){ callback(toBase64(xhr.response)) };
        xhr.onerror = function(){ callback(xhr.status) };
        xhr.open('GET', uri);
        xhr.send();
    ''', uri)
    if type(result) == int :
        raise Exception("Request failed with status %s" % result)
    return base64.b64decode(result)

def download_image(url, file_path, driver):
    bytes = get_file_content_chrome(driver,url)

    # Load image from BytesIO
    im = Image.open(BytesIO(bytes))
    # Display image
    #im.show()
    # Save the image to 'result.FORMAT', using the image format
    im.save(file_path)

def get_image(latitude, longtitude, output_file, driver, first, level=0):
    """
    Function to download image with proper settings from sentinel playground

    Args:
        latitude (int): GPS Latitude
        longtitude (int): GPS Longtitude
        output_file (str): Where to save the downloaded image
        driver (selenium webdriver): Selenium WebDriver instance
        first (bool): Is this first time running ? Do we need to accept TOS ? 
    """
    if(level >= 2):
        return -2
    # Load page 
    driver.get(get_url(latitude,longtitude))
    
    # If first run -> have to accept TOS
    if(first):
        # Wait until Accept TOS button is present  
        try:
            WebDriverWait(driver, 15).until(ec.presence_of_element_located((By.CLASS_NAME, 'accept-btn')))
        except TimeoutException:
            data = get_image(latitude, longtitude, output_file, driver, False, level=level+1)
            if(data == -2):
                return -2
            return -1
        # Accept TOS
        accept_tos(driver)

    # Wait until generate image button is loaded
    try:
        WebDriverWait(driver, 15).until(ec.presence_of_element_located((By.CLASS_NAME, 'CodeMirror-scroll')))
    except TimeoutException:
            data = get_image(latitude, longtitude, output_file, driver, False, level=level+1)
            if(data == -2):
                return -2
            return -1
    generate_image(driver)
    
    # Wait until image generated
    try:
        WebDriverWait(driver, 30).until(ec.visibility_of_element_located((By.ID, 'sentinelImage')))
    except TimeoutException:
        data = get_image(latitude, longtitude, output_file, driver, False, level=level+1)
        if(data == -2):
                return -2
        return -1
    # Get image URL
    url = get_image_url(driver)
    # Download image
    download_image(url, output_file, driver)
    # Close driver instance
    driver.quit()
    return 0

def contrast_stretch(im):
    """
    Makes image more contrased

    Args:
        im (image): Image to contrast

    Returns:
        Image: contrasted image 
    """
    in_min = np.percentile(im, 5)
    in_max = np.percentile(im, 95)

    out_min = 0.0
    out_max = 255.0

    out = im - in_min
    #warnings.filterwarnings('error')
    with np.errstate(divide='raise'):
        try:
            out *= ((out_min - out_max) / (in_min - in_max))
        except:
            #warnings.filterwarnings('ignore')
            return -1
    #warnings.filterwarnings('ignore')
    out += in_min
    return out

def calc_ndvi(image):
    """
    Function that calculates NDVI of image, red channel stays, blue is replaces by NIR, don't care about green

    Args:
        image (image): Image on which NDVI should be calculated

    Returns:
        _type_: _description_
    """
    #np.seterr(all='raise')
    b, g, r = cv2.split(image)
    bottom = (r.astype(float) + b.astype(float)) # Red + NIR
    bottom[bottom==0] = 0.01
    ndvi = (b.astype(float) - r) / bottom
    return ndvi

def process_img(img_path, index):
    """
    Function to process single image (contrast, calculated NDVI...)

    Args:
        img_path (string): path to basic image
        index (int): index of image for saving purposes
    """
    folder = f'output/img{index}'
    # Načtení obrázku
    original = cv2.imread(img_path)

    # Zvýšení kontrastu
    contrasted = contrast_stretch(original)
    if(type(contrasted) != np.ndarray):
        return -1
    # Výpočet NDVI
    ndvi = calc_ndvi(contrasted)
    # Kontrast NDVI
    ndvi_contrasted = contrast_stretch(ndvi)  

    # Color mapping NDVI obrázku pro snazší lidské zpracování
    color_mapped_prep = ndvi_contrasted.astype(np.uint8)
    color_mapped_image_ndvi = cv2.applyColorMap(color_mapped_prep, cv2.COLORMAP_JET)

    # Zapsání výstupních souborů
    cv2.imwrite(f'{folder}/img{index}_ndvi.jpg', ndvi_contrasted)
    cv2.imwrite(f'{folder}/img{index}_c-mapped.jpg', color_mapped_image_ndvi)
    return 0

def worker(data):
    """
    Function that is run on every worker thread

    Downloads image, processes image

    Args:
        data (array): array of input parameters [latitude, longtitude, index]
    """
    t = perf_counter()
    global failed
    # Unpack data from array to separate variables
    latitude = data[0]
    longtitude = data[1]
    index = data[2]
    first_try = data[3]

    # Prepare directory for output data
    try:
        mkdir(f"output/img{index}")
    except:
        global pre_done
        pre_done += 1
        print(f"{text.fg.blue}[N] Image {index}: already done (folder exists){text.reset}")
        logger.info(f"Image {index}: already done (folder exists)")
        return 0
    global processed
    if not first_try:
        processed += 1
    # Define downloaded file
    output_file = f"output/img{index}/img{index}.jpg"
    # Initialize chrome instance
    driver = webdriver.Chrome()
    # Set size of chrome instance to FHD
    driver.set_window_size(1920,1080)
    # Get image using chrome instance
    dat = get_image(latitude, longtitude, output_file, driver, 1)
    if(dat == -2):
        n = notify2.Notification("Failed to process image", message="Failed to process image due to time out while getting base image!", icon="fail.png")
        n.set_timeout(2500)
        n.set_urgency(notify2.URGENCY_LOW)
        n.show()
        failed += 1
        t2 = perf_counter()
        perf = "%(x).2f" % {"x":t2 - t}
        print(f'{text.fg.red}[E] Image {index}: unable to process (timed out) :: {perf}s')
        logger.error(f"Image {index}: unable to process (timed out) :: {perf}s")
        shutil.rmtree(f'output/img{index}')
        rmdir(f'output/img{index}')
        return 0
    
    # Load save image 
    img = Image.open(output_file)
    # If image is only black or only white
    if sum(img.convert("L").getextrema()) in (0, 2):
        global black_out 
        black_out += 1
        # Delete image and folder, clean up after it, print to console, exit function
        t2 = perf_counter()
        perf = "%(x).2f" % {"x":t2 - t}
        print(f"{text.fg.orange}[W] Image {index}: all black [ocean] :: {perf}s{text.reset}")
        logger.warning(f"Image {index}: all black [ocean] :: {perf}s")
        shutil.rmtree(f'output/img{index}')
        rmdir(f'output/img{index}')
        return 0
    # Process img
    val = process_img(output_file, index)
    # Error in image processing, sometimes happens, don't know why, but it's not that often, so it's much easier to just ignore cases when that happens
    if(val == -1):
        n = notify2.Notification("Failed to process image", message="Failed to process image due to error in image processing!", icon="fail.png")
        n.set_timeout(2500)
        n.set_urgency(notify2.URGENCY_LOW)
        n.show()
        failed += 1
        t2 = perf_counter()
        perf = "%(x).2f" % {"x":t2 - t}
        print(f'{text.fg.red}[E] Image {index}: unable to process (error in image processing) :: {perf}s')
        logger.error(f"Image {index}: unable to process (error in image processing) :: {perf}s")
        shutil.rmtree(f'output/img{index}')
        try:
            rmdir(f'output/img{index}')
        except:
            pass
        return 0
    # Edit EXIF data
    info = gpsphoto.GPSInfo((float(latitude), float(longtitude)))
    # NDVI image
    photo = gpsphoto.GPSPhoto(f'output/img{index}/img{index}_ndvi.jpg')
    photo.modGPSData(info, f'output/img{index}/img{index}_ndvi.jpg')
    # Color mapped image
    photo = gpsphoto.GPSPhoto(f'output/img{index}/img{index}_c-mapped.jpg')
    photo.modGPSData(info, f'output/img{index}/img{index}_c-mapped.jpg')
    # Downloaded image  
    photo = gpsphoto.GPSPhoto(output_file)
    photo.modGPSData(info, output_file)

    t2 = perf_counter()
    perf = "%(x).2f" % {"x":t2 - t}
    # Printout
    print(f"{text.fg.green}[S] Image {index}: processed successfuly! Latitude: {latitude}N Longtitude: {longtitude}E :: {perf}s{text.reset}")
    logger.info(f"Image {index}: processed successfuly! Latitude: {latitude}N Longtitude: {longtitude}E :: {perf}s")
  

def scan_check(directory):
    global values
    global rests
    dirs = listdir(directory)
    #size = len(dirs)
    rests = []
    for folder in dirs:
        inside = listdir(f'{directory}/{folder}')
        size_in = len(inside)

        if(size_in != 3):
            tmpArr = values[int(folder[folder.index("g")+1:])]
            tmpArr.append(True)
            rests.append(tmpArr)
            shutil.rmtree(f'output/img{folder[folder.index("g")+1:]}')
            try:
                    rmdir(f'output/img{folder[folder.index("g")+1:]}')
            except:
                pass
    
    if(len(rests) > 0):
        print(f"{text.fg.cyan}[I] Scanned output for errors: {len(rests)}")
        logger.warning(f"Scanned output for errors: {len(rests)}")
        n = notify2.Notification("Scan for error", message=f"Found {len(rests)} errors in output. Fixing!", icon="quest.png")
        n.set_timeout(2500)
        n.set_urgency(notify2.URGENCY_LOW)
        n.show()
    else:
        print(f"{text.fg.cyan}[I] Scanned output for errors: no errors found")
        logger.info(f"Scanned output for errors: no errors found")

if(__name__ == "__main__"):
    # Timer to time how long did program run for
    start_time = perf_counter()
    # Global counters
    global black_out
    global failed
    global processed
    global pre_done
    global rests
    global values
    black_out = 0
    failed = 0
    pre_done = 0
    processed = 0
    rests = []

    # Setup logging
    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': True,
    })
    logging.basicConfig(filename="astropi.log", format='[%(levelname)-8s] %(asctime)-25s %(message)s', filemode='w', level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Settings
    images = 6000   # How many images to try to save
    longtitude = 35 # Longtitude of starting position
    latitude = 14   # Latitude of starting position
    step = 1        # How much change latitude and longtitude between images
    batch_size = 15 # How much concurent processes

    # Prepare array for data output
    values = []
    # Create array of values to later run worker functions on
    for i in range(images):
        values.append([latitude, longtitude, "%(i)04d" %{'i':i}, False])
        latitude += step
        if(latitude > 51):
            latitude = -51
        longtitude += step
        if(longtitude > 180):
            longtitude = -180

    notify2.init("Astro Pi Automated Dataset Creator")

    # Create ThreadPoolExecutor with thread count of batch_size
    with ThreadPoolExecutor(batch_size) as executor:
        # Run on all values from values
        executor.map(worker, values)

    # Check for directories with less than 3 items in them => something happened while processing, redo
    scan_check("output")

    while(len(rests) > 0):
        with ThreadPoolExecutor(batch_size) as executor:
            # Run on all values from rests
            executor.map(worker, rests)
        scan_check("output")


    # Final print out
    #print(f'---------------------\nProcessed {images} images (in batches of {batch_size}) in {perf_counter() - start_time}s!')
    perf = "%(x)2d" % {"x":perf_counter() - start_time}
    print(f"{text.fg.lightgreen}\n{text.bold}------------ DONE ------------")
    print(f"Images to process: {images}")
    print(f"Processed images: {processed}")
    print("---")
    print(f"Number of usable images: {processed - black_out - failed}")
    print(f"Number of thrown away images: {black_out} ")
    print(f"Number of images failed to process: {failed} ")
    print(f"Number of image already done (probably): {pre_done} ")
    print("---")
    print(f"Batch size: {batch_size}")
    print(f"Total time: {perf}s{text.reset}")

    logger.info("")
    logger.info(f"------------ DONE ------------")
    logger.info(f"Images to process: {images}")
    logger.info(f"Processed images: {processed}")
    logger.info("---")
    logger.info(f"Number of usable images: {processed - black_out - failed}")
    logger.info(f"Number of thrown away images: {black_out} ")
    logger.info(f"Number of images failed to process: {failed} ")
    logger.info(f"Number of image already done (probably): {pre_done} ")
    logger.info("---")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Total time: {perf}s")

    n = notify2.Notification("Image processing done!", message=f"Processed {processed} images, more information available in the terminal window!", icon="done.png")
    n.set_urgency(notify2.URGENCY_CRITICAL)
    n.show()

