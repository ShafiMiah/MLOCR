# Text recognition using machine learning
This MLOCR will recognize any text in an image and will extract the text content. It will print the YOLO format label for the co-ordinate of the text in the image. It is also possible to view the classification or text identification result in a viewer. Text region classification can be added manually. I shall show you the step by step procedure how to run the MLOCR using the exe without having headace about code. But I shall describe a little about code.  

## How to Run the Software
You don’t need to install Python or any dependencies to run the software. 
   ```sh
# 1. Go to the dist folder
#    (e.g., ODAI\dist)

# 2. Download the dist.zip and extract to your local disk

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




