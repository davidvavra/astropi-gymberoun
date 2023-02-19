# AI model creation
This folder contains some tools used for creating dataset, creating, training, testing, evaluating and converting model. Everything is done from one file - main.py  
Installation of necessary libraries should be as easy as running `pip install -r requirements.txt`. Due to usage of some older version of some libraries I recommend using python 3.9 or bit older (3.7 or 3.8).    
Then you have to just run the main.py file and set everything to your liking.

    usage: main.py [-h] --mode {dataset,train,evaluate,convert,test} [--color_conversion_map COLOR_CONVERSION_MAP] [--dataset_max_thread DATASET_MAX_THREAD] [--train_epochs TRAIN_EPOCHS] [--image_size IMAGE_SIZE]
                [--batch_size BATCH_SIZE] [--test_image TEST_IMAGE] [--model_path MODEL_PATH] [--save_path SAVE_PATH] [--conversion_size CONVERSION_SIZE] [--expand_dataset EXPAND_DATASET]
                [--model {UNET,UNet,DeepLabV3_plus,DLABV3_plus,FCN,PAN}] [--backbone {VGG16,VGG19,MobileNetV1,MobileNetV2}] [--num_classes NUM_CLASSES] [--optimizer {Adam,Adagrad,SGD}] --base_dataset_dir
                BASE_DATASET_DIR [--colormap COLORMAP]

    Astropi AI creation tool for semantic segmentation models Tool for creating dataset, creating, evaluating and testing semantic segmentation models and model conversion to tflite INT8 fully quantized model almost
    ready to be run on EdgeTPU

    optional arguments:
    -h, --help            show this help message and exit
    --mode {dataset,train,evaluate,convert,test}
                            Choose which mode to use.
    --color_conversion_map COLOR_CONVERSION_MAP
                            Only for 'dataset' mode. Path to file with color conversion map.
    --dataset_max_thread DATASET_MAX_THREAD
                            Maximum thread to run when creating dataset. Only for mode 'dataset'
    --train_epochs TRAIN_EPOCHS
                            How many training epochs to run. Only for 'train'.
    --image_size IMAGE_SIZE
                            Image must be square, size of side in pixels. For all modes except 'dataset' and 'convert'.
    --batch_size BATCH_SIZE
                            How big should batch_size be. For modes: 'evaluate', 'train', 'convert'
    --test_image TEST_IMAGE
                            If test mode is selected show inference with this image (image will be shown)
    --model_path MODEL_PATH
                            Path to saved model. Only for modes ['test','evaluate','convert']
    --save_path SAVE_PATH
                            Path to folder where models should be saved. Only for 'training' or 'convert'.
    --conversion_size CONVERSION_SIZE
                            Size of representative dataset for conversion, remember, conversion to tflite with full INT8 quantization is pretty lenghty process, recommended value is ~100 images. Will be selected
                            randomly from validation set, -1 will use all images from validation, -2 will use all images. Only if mode 'convert'
    --expand_dataset EXPAND_DATASET
                            Automatically expand dataset. Good if dataset is too small. Only if mode is 'dataset'
    --model {UNET,UNet,DeepLabV3_plus,DLABV3_plus,FCN,PAN}
                            Choose semantic segmentation model to use.
    --backbone {VGG16,VGG19,MobileNetV1,MobileNetV2}
                            Choose backone model.
    --num_classes NUM_CLASSES
                            The number of classes to be segmented.
    --optimizer {Adam,Adagrad,SGD}
                            Optimizer to use for the model.
    --base_dataset_dir BASE_DATASET_DIR
                            Base folder where the images (dataset) are located.
    --colormap COLORMAP   Path to file containing colormap to recolor images after inferencing. Only if mode is 'test'

As you can see, you can create dataset or generate, train, evaluate, test and convert model.

## Step-by-step tutorial
Here is step-by-step tutorial on using this script to create your own semantic segmentation model that is capable of running on EdgeTPU device.

### Generating dataset
To generate dataset you have to firstly prepare your images. Each image must have it's corresponding mask. Program is expecting you to have training and validation images.  
You simply create a folder with two subfolder: `train` and `validation`  
You file/folder structure should look like this:

    dataset
    ├── train
    │   ├── images
    │   │   ├── train_img1.png
    │   │   ├── train_img2.png
    │   │   ├── train_img3.jpg
    │   │   ├── train_img4.png
    │   │   ├── train_img5.png
    │   │   ├── train_img6.png
    │   │   ├── train_img7.png
    │   │   └── train_img8.png
    │   └── masks
    │       ├── train_img1.png
    │       ├── train_img2.png
    │       ├── train_img3.jpg
    │       ├── train_img4.png
    │       ├── train_img5.png
    │       ├── train_img6.png
    │       ├── train_img7.png
    │       └── train_img8.png
    └── validation
        ├── images
        │   ├── validation_img1.png
        │   ├── validation_img2.png
        │   ├── validation_img3.png
        │   ├── validation_img4.png
        │   ├── validation_img5.png
        │   ├── validation_img6.jpg
        │   ├── validation_img7.png
        │   └── validation_img8.png
        └── masks
            ├── validation_img1.png
            ├── validation_img2.png
            ├── validation_img3.png
            ├── validation_img4.png
            ├── validation_img5.png
            ├── validation_img6.jpg
            ├── validation_img7.png
            └── validation_img8.png

*NOTE: it's important that images and masks have exactly same names, also note that files can be png or jpg as long as it's same for both. You can also combine png and jpg files.*  
Command for generating dataset will look something like this `python main.py --mode dataset --base_data_dir /path/to/dataset --dataset_max_thread 20 --expand_dataset y --color_conversion_map map.txt`  
This command will generate dataset from provided images, parameter `--dataset_max_thrad` limits how many threads can this process use, it's recommended to leave at least two thread  
`--expand_dataset` argument is used to select if you want to expand your dataset by rotating each images by 2.5° until full rotation is achieved and saving each image along the way. This will make your dataset considerably bigger (~144 time bigger). This is useful if you don't have enough images.  
Probably most important argument is `--color_conversion_map`. Training the model requires you to have masks in B&W, but it's very hard for humans to use images like that so you want to provide color converion map. It's a file containing HSV ranges for each color in your dataset and corresponding B&W color.  
Sample *map.txt*

    0,0,0;20,255,255;5
    20,0,0;50,255,255;1
    50,0,0;60,255,255;2
    59,0,0;75,255,255;3
    75,0,0;105,255,255;4
    105,0,0;118,255,255;0
    118,0,0;130,255,255;6
    130,0,0;140,255,255;7
    140,0,0;170,255,255;8
    170,0,0;180,255,255;9
    180,0,0;255,255,255;0

Values are divided by command and semicolons. First 3 values (divided by commas) are bottom HSV border, second 3 values are top HSV border and last value is RGB value. You want to put semicolon after first 3 values and after second 3 values (`H1,S1,V1;H2,S2,V2;BW`). Also it's recommended to fill up entire HSV spectrum ([0,0,0] to [255,255,255]) and make sure you are not skipping any BW values, start from 0 and go up, you can asign more HSV limits to one BW value (in this example 0 is unknown, so there is category for unknown and everything that doesn't fit will also be asigned 0).


### Training model
Now that you have your dataset prepared you can start training your model. First you need to pick which model do you want, there are many great resources documenting how to pick which model do you want. Here you have some models that are implemented here and backbones that you can use for them:

Model | Name | Backbones
---|---|---
UNet | UNet | VGG16, VGG19, MobileNetV1, MobileNetV2
UNET | UNET | None
DeepLab V3 plus | DeepLabV3_plus | VGG16, VGG19, MobileNetV1, MobileNetV2
DeepLab V3 plus | DLABV3_plus | None
FCN | FCN | VGG16, VGG19, MobileNetV1, MobileNetV2
PAN | PAN | VGG16, VGG19, MobileNetV1, MobileNetV2

*NOTE: Model implementations without availible backbones are implementation written by me and they work without it (either they use just one or the don't need it at all), other implementations are from https://github.com/luyanger1799/amazing-semantic-segmentation*
As of optimizers you can choose between `SGD`, `Adagrad` and `Adam`.  
Example command to train model would look like this `python main.py --mode train --train_epochs 50 --image_size 1024 --batch_size 1 --save_path output --model PAN --backcbone VGG16 --num_classes 10 --optimizer Adam --base_data_dir /path/to/dataset`  
`--train_epochs 50` sets the model to train for 50 epochs. `--image_size 1024` sets image resolution of model inputs to be 1024x1024 pixels, if images & masks aren't of this size it doesn't matter, they will be scaled down (but keep in mind that it may harm model's performance). `--batch_size 1` sets the batch size to be, this is because on my computer I can't fit more images into my GPU's RAM, but if you have more GPU RAM (>8GB) that you may put increase the batch size and it will make model learn faster. `--save_path output` will create folder name *output* in the same directory as main.py is located and it will put all saved models into that folder. After that we say which model we want (PAN with VGG16 backbone, optimizer Adam). `--num_classes 10` is I think pretty self-explanatory, it just tells the code how many classes we are segmenting images into. Last argument is path to our dataset directory.
*NOTE: Model training is computationaly expensive process and it may take a lot of time. On RTX 3070 Ti it took training this exact model on dataset of almost 20k images more than 24 hours.*

### Evaluating model  
Evaluating the model may be useful especially if you've created more models and you want to compare their performances on the same data. For evaluation is generally best to use different dataset than used for training and/or validation, but since we were limited to small dataset we've evaluated models on same datasets. Remember that you have to give same model parameters as you gave while training, this program recreates the model and then loads weights from the file.  
Example command to evaluating on same dataset could look like this `python main.py --mode evaluate --model PAN --backbone VGG16 --num_classes 10 --optimizer Adam --image_size 1024 --batch_size 1 --base_data_dir /path/to/dataset --model_path output/model.hdf5`  
As you can see, the command is very similar to previous command, the main difference is that we provide path to save .hdf5 (or .h5) model and use `evaluate` mode instead of train mode. This process wil run for quite some time, but it won't run for time similar to training. Evaluating our model took aproximately 20 minutes (remember that training took more than 24 hours).  
At the end of the process you will get two numbers: `loss` and `accuracy`, you are probably going to be more interested in accuracy, as it tells you how much correctly colored pixels on that dataset did your model produce (it's in %, so 0.74 is 74%).  

### Testing model
You may also want to see how your model performs with your own eyes instead just reading and believing a bunch of numbers. For that there is *test* mode. Usage of that command is practicaly same as of the evaluate, but you don't need to supply `--batch_size` argument and `--base_data_dir` arguments, you need to supply `--test_image` argument. This should be a path to image you want to test your model on, it can be image from your dataset or any other image. It will run the image through the model and show you, how the model processed that image.  
You also  need to provide `--colormap`, which is path to file containing colormap. Colormap is used for showing you what the model predicts in more human-friendly colors than 10 shades of black. File containing colormap should look like this:  

    255,106,0
    255,255,0
    7,89,0,
    7,255,0
    0,255,215
    90,106,220
    0,0,215
    67,0,129
    227,181,215
    255,106,129

*NOTE: These colors are in RGB spectre. Also colors to B&W colors are asigned as they go, so first line will be asigned to B&W color 0, second line to 1, etc. Make sure that you have same amount of lines in colormap as classes that you are segmenting into. Also values here are divided only by commas and are in order Red, Green, Blue.*

### Converting model
If you want to run your model on EdgeTPU device you will need to convert it to .tflite format and quantize it to use INT8. That is made easy using this script, you will just select `convert` mode and otherwise copy command for evaluation. To that you will need to add `--conversion_size` parameter. This decides how many images are you going to use as representative dataset, these are randomly selected pictures from your validation dataset. You can also set it's value to -1 which would use all validation images and -2 which would use all images (even training). You also don't need `--batch_size` argument.  
Keep in mind, that this process can also take a few minutes (more like half an hour) so you may want to choose `--conversion_size` wisely, I'd recommend somewhere in the neighborhood of 100 images. If you've augmented your dataset you may also consider using only original images.  You also need to provide `--save_path` argument like in model training.  
#### EdgeTPU compilation
After model conversion using this script you want to navigate to https://colab.research.google.com/github/google-coral/tutorials/blob/master/compile_for_edgetpu.ipynb#scrollTo=joxrIB0I3cdi in your browser. This is google colab notebook that let's you compile model to be edgetpu compatible. You upload your .tlfite model there and rest you just follow instructions. If compilation for some reason fails try adding `-s -d` to `edgetpu_compiler $TFLITE_FILE` compiler command (`edgetpu_compiler -s -d $TFLITE_FILE`). Otherwise you may want to use google and maybe try different settings in model training/converting process :D.
#### Running the model
How to run the model is described for example here https://github.com/tensorflow/examples/tree/master/lite/examples/image_segmentation/raspberry_pi