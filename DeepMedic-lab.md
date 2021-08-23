
# Table of Contents
- [How to install DeepMedic on Ubuntu?](#How to install DeepMedic on Ubuntu?)

- [How to use the pre-trained model to segment brain tumors?](#How to use the pre-trained model to segment brain tumors?)

- [How can I use my local data to train the  model?](#How can I use my local data to train the  model?)

- [How can I prepare my data?](#How can I prepare my data?)


#How to install DeepMedic on Ubuntu?

Before starting the installation you will need to check if you have cuda and the graphics card drivers already setup on your machine. You can do this using these commands:

```bash
#get cuda version
nvcc --version
#Check GPU info
nvidia-smi
#find the path to cuda
whereis -b cuda
```

If you get an output that looks like this, great you have [cuda](https://developer.nvidia.com/cuda-downloads) installed on your machine and you can continue this tutorial, otherwise you can find [here](https://docs.vmware.com/en/VMware-vSphere-Bitfusion/3.0/Example-Guide/GUID-ABB4A0B1-F26E-422E-85C5-BA9F2454363A.html) the method to install cuda.
```[mohamedmakhlouf@fedora nnUNet]$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2021 NVIDIA Corporation
Built on Wed_Jun__2_19:15:15_PDT_2021
Cuda compilation tools, release 11.4, V11.4.48
Build cuda_11.4.r11.4/compiler.30033411_0
```

Another package required by the program and that we must check before starting the installation is [TensorFlow](https://www.tensorflow.org/?hl=fr) do this you must use the command below, if it displays a version higher than 2.0 it's good you have TF installed. 
```bash
#check Tf installed version
python3 -c 'import tensorflow as tf; print(tf.__version__)'

#Expected output style => 2.0
2.6.0
```
if not I invite you to Download it from [here](https://www.tensorflow.org/install?hl=fr) by choosing the right version compatible with your GPU. 
```bash
#example of command for Linux
pip install tensorflow-gpu==2.0
```

From now on you will open a terminal where you want to install the framework and download the software with these commands
```bash
git clone https://github.com/Kamnitsask/deepmedic/
```


Then you have to configure a virtual environment for python using conda which will collect all the packages required by the framework using these commands.
```bash
#Dont forget to replace FOLDER_FOR_ENVS by the Folder of your environement if u have one or u can use this command to create one
mkdir FOLDER_FOR_ENVS  #IF YOU WANT TO USE THE SAME FOLDER NAME 
conda create -p FOLDER_FOR_ENVS/ve_tf_dmtf python=3.6.5 -y
source activate FOLDER_FOR_ENVS/ve_tf_dmtf
```
Once you have setup and activated your conda virtual environment correctly you will see that your prompt has changed, now you have to go to the root of the deepMedic project that you cloned a little before to install all the dependencies with these commands:

```bash
$ cd DEEPMEDIC_ROOT_FOLDER
$ pip install .

```

To be able to use the model with 3D CNN architecture you need to ensure that TensorFlow is  able to find CUDA’s compiler, you can do this quickly following these instructions:

```bash
#check if this variables exist already in the environment
echo $CUDA_HOME
echo $LD_LIBRARY_PATH
echo $PATH
#if you have 3 lines with paths to cuda and cuda's libraries you have nothing to do
#else 
#If you want a solution to do it once you have to adapt and add the last three lines here  to your  .bachrc file using this commands
cd 
sudo nano .bashrc

#Or you can just tape this 3 commands before using DeepMedic in the same terminal that you would use after to manipulate nnUnet. 
#This is a temporory solution if you do not have  access to root
$ export CUDA_HOME=/path/to/cuda                   # If using cshell instead of bash: setenv CUDA_HOME /path/to/cuda
$ export LD_LIBRARY_PATH=/path/to/cuda/lib64
$ export PATH=/path/to/cuda/bin:$PATH

```
**PS: To get the path of cuda check the top of this document**

And finally to get acceleration and have fast segmentation you have to check if cuDNN is installed with this command, if you don't get the same output you can easily install it with these commands:
```bash
#To check if cuDNN is already installed (REPLACE PATH/TO/CUDA)
cat /path/to/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
#if you have something like this as output  you have cuDNN installed before
$ cat /usr/local/cuda-11.4/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
#define CUDNN_MAJOR 8
#define CUDNN_MINOR 2
#define CUDNN_PATCHLEVEL 2
--
#define CUDNN_VERSION (CUDNN_MAJOR * 1000 + CUDNN_MINOR * 100 + CUDNN_PATCHLEVEL)

#endif /* CUDNN_VERSION_H */

#else download cudNN from here : https://developer.nvidia.com/cudnn
#copy the cudnn.h file
sudo cp ./cuda/include/cudnn.h /usr/local/cuda/include
#Copy all files under cuda/lib64/to the/usr/local/cuda/lib64 folder and add read permissions:
sudo cp ./cuda/lib64/* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
#reboot your machine and check if cuDNN is already installed using command above
```

To test your installation you can run this command if you have an output that looks like this and nothing crash it means that you have succeeded in your installation.
```bash
#You have to be in the root of deepMedic and your virtual environment is activated
#run this command to check that the model is working on cpu correctly
./deepMedicRun -model ./examples/configFiles/tinyCnn/model/modelConfig.cfg \
               -train examples/configFiles/tinyCnn/train/trainConfigWithValidation.cfg
#it may takes a few minutes depending on your CPU capacities and the end of the output should appeared like this

#TIMING: Training process lasted: 141.9 secs.
#Closing worker pool.
#Saving the final model at:/home/mohamedmakhlouf/deepmedic/examples/output/saved_models//trainSessionWithValidTiny//tinyCnn.trainSessionWithValidTiny.final.2021-08-20.16.41.33.969777
#The whole do_training() function has finished.

#=======================================================
#=========== Training session finished =================
#=======================================================
#Finished.

# Training board  check
python plotTrainingProgress.py examples/output/logs/trainSessionWithValidTiny.txt -d
tensorboard --logdir=./examples/output/tensorboard/trainSessionWithValidTiny
#A plot should appear in the end 
#GPU check
./deepMedicRun -model ./examples/configFiles/tinyCnn/model/modelConfig.cfg \
               -train ./examples/configFiles/tinyCnn/train/trainConfigWithValidation.cfg \
               -dev cuda0
#if the output of this command looks like this, congratulations you did a great job !  you have a correct installation
            
# ##########################################################################################
##		  Finished full Segmentation of Validation subjects   			#
###########################################################################################
#=============== LOGGING TO TENSORBOARD ===============
#Logging validation metrics from segmentation of whole validation volumes.
#Epoch: 1
#Step number (index of subepoch since start): 3
#Logged metrics: ['whole scans: Dice1 (Prediction VS Truth)', 'whole scans: Dice2 (Prediction within ROI mask VS Truth)', 'whole scans: Dice3 (Prediction VS Truth, both within ROI mask)']
#======================================================
#TIMING: Training process lasted: 98.1 secs.
#Closing worker pool.
#Saving the final model at:/home/mohamedmakhlouf/deepmedic/examples/output/saved_models//trainSessionWithValidTiny//tinyCnn.trainSessionWithValidTiny.final.2021-08-20.16.47.59.242390
#The whole do_training() function has finished.

#=======================================================
#=========== Training session finished =================
#=======================================================
#Finished.

```

#How to use the pre-trained model to segment brain tumors?
It is possible to use the model just to segment tumors, but before doing that you need to design a model with the configuration file and train the model first to find the right parameters. For more details on how to run the training on your database click here.

Suppose you have already a pre trained model, that you can use it to segment tumors by following one of these two methods.

```bash
#1 A model is specified straight from the command line
./deepMedicRun -model ./examples/configFiles/deepMedic/model/modelConfig.cfg \
               -test ./examples/configFiles/deepMedic/test/testConfig.cfg \
               -load ./path-to-saved-model/filename.model.ckpt \
               -dev cuda0
#2 Or the path to a saved model can be instead specified in the testing config file, and then the -load option can be ommited. Note: A file specified by -load option overrides any specified in the config-file.           
```
After the model is loaded, inference will be performed on the testing subjects. Predicted segmentation masks, posterior probability maps for each class, as well as the feature maps of any layer can be saved. If ground-truth is provided, DeepMedic will also report DSC metrics for its predictions.

The test configuration file contains several testing settings which you can find detailed information about here, including examples for image segmentation here.


**Testing Parameters**

*Main Parameters:*

- sessionName: The name for the session, to use for saving the logs and inference results.
- folderForOutput: The output folder to save logs and results.
- cnnModelFilePath: The path to the cnn model to use. Disregarded if specified from command line.
- channels: List of paths to the files that list the files of channels per testing case. Similar to the corresponding parameter for training.
- namesForPredictionsPerCase: Path to a file that lists the names to use for saving the prediction for each subject.
- roiMasks: If masks for a restricted Region-Of-Interest can be made, inference will only be performed within it. If this parameter is omitted in the config file, whole volume is scanned.
- gtLabels: Path to a file that lists the file-paths to Ground Truth labels per case. Not required for testing, but if given, DSC accuracy metric is reported.

*Saving Predictions:*

- saveSegmentation, saveProbMapsForEachClass : Specify whether you would like the segmentation masks and the probability maps of a class saved.


#How can I use my local data to train the  model?

The **.cfg configuration files** in `examples/configFiles/deepMedic/` provides parameters for creating and training DeepMedic. These parameters are similar (but not same) as what was used in our work in [[1](#citations)] and our winning contribution for the ISLES 2015 challenge [2]. In order to support a broader range of applications and users, the config files in `examples/configFiles/deepMedic/` are gradually updated with components that seem to improve the overall performance of the system. (Note: Original config as used in the mentioned papers can be found in archived github-branch 'dm_theano_v0.6.1_depr')

To run the DeepMedic on your data, the following are the minimum steps you need to follow:

**a)** **Pre-process your data** as described in Sec. [1.4](#14-required-data-pre-processing). Do not forget to normalise them to a zero-mean, unit-variance space. Produce ROI masks (for instance brain masks) if possible for the task.

**b)** In the **modelConfig.cfg** file, change the variable `numberOfOutputClasses = 5` to the number of classes in your task (eg 2 if binary), and `numberOfInputChannels = 2` to the number of input modalities. Now you are ready to create the model via the `-newModel` option.

**c)** (optional) If you want to train a bigger or smaller network, the easiest way is to increase/decrease the number of Feature Maps per layer. This is done by changing the number of FMs in the variable `numberFMsPerLayerNormal = [30, 30, 40, 40, 40, 40, 50, 50]`.

**d)** Before you train a network you need to alter the **trainConfig.cfg** file, in order to let the software know where your input images are. The variable `channelsTraining = ["./trainChannels_flair.cfg", "./trainChannels_t1c.cfg"]` is pre-set to point to two files, one for each of the input variables. Adjust this for your task. 

**e)** Create your files that correspond to the above `./trainChannels_flair.cfg, trainChannels_t1c.cfg` files for your task. Each of these files is essentially a list. Every file has an entry for each of the training subjects. The entry is the path to the .nii file with the corresponding modality image for the subject. A brief look to the provided exemplary files should make things clear.

**f)** Do the same process in order to point to the ground-truth labels for training via the variable `gtLabelsTraining = "./trainGtLabels.cfg"` and to ROI masks (if available) via `roiMasksTraining = "./trainRoiMasks.cfg"`.

**g)** If you wish to periodically perform **validation** throughout training, similar to the above, point to the files of validation subjects via the variables `channelsValidation`, `gtLabelsValidation` and `roiMasksValidation`. If you do not wish to perform validation (it is time consuming), set to `False` the variables `performValidationOnSamplesThroughoutTraining`
and `performFullInferenceOnValidationImagesEveryFewEpochs`.

**h)** (optional) If you need to adjust the length of the training session, eg for a smaller network, easiest way is to lower the total number of epochs `numberOfEpochs=35`. You should also then adjust the pre-defined schedule via `predefinedSchedule`. Another option is to use a decreasing schedule for the learning rate, by setting `typeOfLearningRateSchedule = 'poly'`.

**i)** **To test** a trained network, you need to point to the images of the testing subjects, similar to point d) for the training. Adjust the variable `channels = ["./testChannels_flair.cfg", "./testChannels_t1c.cfg"]` to point to the modalities of the test subjects. If ROI masks are available, point to them via `roiMasks` and inference will only be performed within the ROI. Else comment this variable out. Similarly, if you provide the ground-truth labels for the testing subjects via `gtLabels`, accuracy of the prediction will be calculated and the DSC metric will be reported. Otherwise just comment this variable out.

**j)** Finally, you need to create a file, which will list names to give to the predictions for each of the testing subject. See entry `namesForPredictionsPerCase = "./testNamesOfPredictionsSimple.cfg"` and the corresponding pre-set file. After that, you are ready to test with a model.

The provided configuration of the DeepMedic takes roughly 2 days to get trained on an NVIDIA GTX Titan X. Inference on a standard size brain scan should take 2-3 minutes. Adjust configuration of training and testing or consider downsampling your data if it takes much longer for your task.


#Config used 
https://github.com/deepmedic/deepmedic/blob/dm_theano_v0.6.1_depr/examples/configFiles/deepMedic/model/modelConfig.cfg