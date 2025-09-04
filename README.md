# Text recognition using machine learning
This MLOCR will recognize any text in an image and will extract the text content. It will print the YOLO format label for the co-ordinate of the text in the image. It is also possible to view the classification or text identification result in a viewer. Text region classification can be added manually. I shall show you the step by step procedure how to run the MLOCR using the exe without having headace about code. But I shall describe a little about code.  
## Requirements
You need to install Python 3.12.2 or higher. Install torch correct version according to your GPU support. 
Install all the required packages listed in package.txt. Then run or double click the script CreateExe.sh. It will generate dist folder in your working directory.

## How to Run the Software
You don’t need to install Python or any dependencies to run the software. 
   ```sh
# 1. Go to the dist folder
#    (e.g., ODAI\dist)

# 2. Get the dist and extract to your local disk

# 3. Open Command Prompt (or a terminal)

# 4. Navigate to the copied dist folder
cd pathtolocal\dist

# 5. Run the executable and initialize the software
Main_ODAI.exe --operation init
```

# Settings
Go to ~dist\Config path and open settings.xml. We have following setting
```sh
<settings>
	<add key="SourceImageDirectory" value="C:\Data\SourceImages" />
	<add key="IllustrationImageNamesContainerFile" value="C:\Data\images.csv" />
	<add key="TrainImageDirectory" value="C:\Data\images\train" />
	<add key="LabelDirectory" value="C:\Data\labels\train" />
	<add key="ClassFileDirectory" value="SMLabel\data\predefined_classes.txt" />
	<add key="OutputDirectory" value="C:\Temp" />
	<add key="Regex" value="^\d+[A-Za-z]*$" />
	<add key="NumberOfEpoch" value="100" />
</settings>
```
**SourceImageDirectory :** The image location which you will use for your dataset.

**IllustrationImageNamesContainerFile:** It may contain a subset of image names or all images names which will be used for ML training.

**TrainImageDirectory:** The image directory where the image will be placed for training purpose. Make sure that it follows {Base directory}\images\train. 

**LabelDirectory:** The annotation of the images will be kept here.  Make sure that it follows {Base directory}\labels\train. Note that for TrainImageDirectory and LabelDirectory - base directoy must be same.

**ClassFileDirectory:** You point out the class definition file. As we are creating a ML for text detection then the class name will be only **Text**. But if you want it for other ML detection model then on class file each class name will be in a new line.

**OutputDirectory:** The directory where the prediction result will be printed out in text file.

**Regex:** You can put regex so that you can identify and get the text in the images which you are interested in.

**NumberOfEpoch:** How many epoch YOLOv8 model will run.
# Override settings
Note that you can always override the settings value by providing the argument when you run the process to complete a certain types of operation. Here are the parameters description.
```sh
"--operation",  required=True, help="Type of operation: TransferCSVImage,AutoAnnotation,PredictHotspot"
"--imagepath",  help="This is the source image path"
"--trainpath",  help="This is the train image path"
"--labelpath",  help="This is the train annotation path"
"--classfile",  help="This is the class list text file"
"--imagenamescontainerfile",  help="This is contains all the names of images which used for training"
"--outputpath",  help="This is the path where the prediction annotationa and class will be written"
"--regex",  help="This will filter out the text with regex"
"--display", type=str2bool, nargs="?", const=True, default=False, help="Set True or False to show/annotate on viewer"
```
# 1. Transfer subset of images for training 
You can define the csv file which will contain all images names seperated by comma which will be read images from the SourceImageDirectory.
```sh
Main_ODAI.exe --operation TransferCSVImage --imagepath "C:\SourceImages" --trainpath "SMLabel\data\images\train" --imagenamescontainerfile "c:\imagenames.csv"
```
# 2. Automatically annotate the text area in the image 
```sh
Main_ODAI.exe --operation AutoAnnotation  --trainpath "SMLabel\data\images\train" --labelpath "SMLabel\data\labels\train" --classfile "SMLabel\data\predefined_classes.txt"
```
# 3. Manually annotate the text region
Note that a good training dataset will give you highly performant and accurate model. It is better to check if annotation went correctly and annotate the images properly.
```sh
Main_ODAI.exe --operation ManualAnnotation  --trainpath "SMLabel\data\images\train" --labelpath "SMLabel\data\labels\train" --classfile "SMLabel\data\predefined_classes.txt"
```
# 4. Create a validation set.
Now, training dataset has been created. The next step is to create a validation dataset.
```sh
Main_ODAI.exe --operation CreatevalidationSet  --trainpath "SMLabel\data\images\train" --labelpath "SMLabel\data\labels\train"
```
# 4. Start training the text detection ML.
```sh
Main_ODAI.exe --operation TrainML  --trainpath "SMLabel\data\images\train" --labelpath "SMLabel\data\labels\train" --classfile "SMLabel\data\predefined_classes.txt"
```
# 5. Prediction of HotSpot in the image for identifying text
The prediction output will be writen in the **OutputDirectory** with the same name of image in YOLO format eg imagetopredict.txt and class number to name presentation in {imagetopredict}classes.txt. Then you can use these 2 file to get the co-ordinate of text hotspot and hotspot name representation.
```sh
Main_ODAI.exe --operation PredictHotspot  --imagepath "C:\SourceImages\imagetopredict.jpg" --regex "^\d+[A-Za-z]*$" --display 
```
# 6. Single Image Annotation
Instead of annotating whole folder image manually- you can annotate only a single image manually.
```sh
Main_ODAI.exe --operation SingleImageAnnotation  --imagepath "C:\SourceImages\imagetopredict.jpg" --labelpath "SMLabel\data\labels\train" --classfile "SMLabel\data\predefined_classes.txt"
```
