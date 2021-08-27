# nnU-net lab
This document contains practical examples of how to use nnUnet to do automatic brain tumor segmentation.
You will find instructions on how to start setting it up until you design a model trained on your own data.

The official documentation of the framework is available [here](https://github.com/MIC-DKFZ/nnUNet), as well as a [shortened-documentation](nnU-Net.md) is available on this project.

To facilitate the use of this document you will find a summary here which contains different scenarios of use of the framework.

# Table of Contents
- [How to install nnUnet on Ubuntu?](#How-to-install-nnUnet-on-Ubuntu?)
- [How to use the pre-trained model to segment brain tumors](#How-to-use-the-pre-trained-model-to-segment-brain-tumors)
- [How can I use my local data to try to improve the pre-trained model?](#How-can-I-use-my-local-data-to-try-to-improve-the-pre-trained-model)
- [How can I prepare my data ?](#How-can-I-prepare-my-data)


## How to install nnUnet on Ubuntu?

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

Another package required by the program and that we must check before starting the installation is [Pytorch](https://pytorch.org/) do this you must use the command below, if it displays a version higher than 1.6 it's good you have TF installed. 
```bash
python3 -c 'import torch; print(torch.__version__)'

#Expected output style
1.9.0+cu111
```
if not I invite you to Download it from [here](https://pytorch.org/get-started/locally/) by choosing the right version compatible with your os and the cuda version and using conda for the install. 
```bash
#example of command for Linux -cuda 10.2 - Stable version 1.9.0 - Conda - Python
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```

From now on you will open a terminal where you want to install the framework and download the software with these commands
```bash
git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet
pip install -e
```


once done you will have to define three path variables using one of these two solutions (do not forget to adapt your paths), you can find more informations about those three paths [here](setting_up_paths.md).
```bash
#If you want a solution to do it once you have to add the last three lines here  to your  .bachrc file using this commands
cd 
sudo nano .bashrc

#Or you can just tape this 3 commands before using nnUnet in the same terminal that you would use after to manipulate nnUnet. 
#This is a temporory solution if you do not have  access to root
export nnUNet_raw_data_base="/media/fabian/nnUNet_raw"
export nnUNet_preprocessed="/media/fabian/nnUNet_preprocessed"
export RESULTS_FOLDER="/media/fabian/nnUNet_trained_models"

#Another solution for isc users, try just to run setUpPaths using this command:
./scriptSETUPpaths
#After that you will be able to configure the variables by entering the three paths
#Examples and explanations will also be displayed on the terminal by running the above command
```

You are almost done with the installation, if you want to visualize the drives you can install this package with this command:
```bash
pip install --upgrade git+https://github.com/FabianIsensee/hiddenlayer.git@more_plotted_details#egg=hiddenlayer

```

To test your installation you can run this command if you have an output that looks like this it means that you have succeeded in your installation.
```bash
#Instalation check
nnUNet_print_available_pretrained_models
#expected output 
[mohamedmakhlouf@fedora ~]$ nnUNet_print_available_pretrained_models


Please cite the following paper when using nnUNet:

Isensee, F., Jaeger, P.F., Kohl, S.A.A. et al. "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation." Nat Methods (2020). https://doi.org/10.1038/s41592-020-01008-z


If you have questions or suggestions, feel free to open an issue at https://github.com/MIC-DKFZ/nnUNet

The following pretrained models are available:


Task001_BrainTumour
Brain Tumor Segmentation. 
Segmentation targets are edema, enhancing tumor and necrosis, 
Input modalities are 0: FLAIR, 1: T1, 2: T1 with contrast agent, 3: T2. 
Also see Medical Segmentation Decathlon, http://medicaldecathlon.com/

Task002_Heart
Left Atrium Segmentation. 
Segmentation target is the left atrium, 
Input modalities are 0: MRI. 
Also see Medical Segmentation Decathlon, http://medicaldecathlon.com/
........
```

## How to use the pre-trained model to segment brain tumors
Several pre-trained models are available, you can see the list of these models at the end of the installation step.
To use the model that allows you to do segmentation of brain tumors try this commands:
```bash
#Display all available models using this
nnUNet_print_available_pretrained_models
#You can then download models by specifying their task name
nnUNet_download_pretrained_model Task001_BrainTumour
#You can display the details of the model using these two methods
#First One ( using nnUnet command)
nnUNet_print_pretrained_model_info Task001_BrainTumour
#Second One (the content of .json file which contains information)
#You have to replace the path to your data_raw_floder
cat nnUNet_raw_data/Task001_BrainTumour
```



To make a segmentation as you normally know the nnUnet has 3 different architectures that can be used separately or to combine the results of several models to have a more sophisticated segmentation. You will find in the following how to do each of these two segmentation methods with a pre-trained model.


If you want to do a prediction without  ensembling, otherwise to use only one architecture for a segmentation, the best one for example the framework allows you to find it easily with this command which will even return you the command to use to make a segmentation with the best architecture.  
Remember that the data located in the input folder must adhere to the format specified [here](data_format_inference.md).
```bash
#returns the best model between the thees 4 models
#use the option -h for more information

nnUNet_find_best_configuration -m 2d 3d_fullres 3d_lowres 3d_cascade_fullres -t 001 --strict

#nnUNet_find_best_configuration will print a string to the terminal with the inference commands you need to use. The easiest way to run inference is to simply use these commands.

```


If you wish to manually specify the configuration(s) used for inference, For each of the desired configurations, run:
```bash
#Only specify --save_npz if you intend to use ensembling.
# --save_npz will make the command save the softmax probabilities alongside of the predicted segmentation masks requiring a lot of disk space.
#Please select a separate OUTPUT_FOLDER for each configuration!
#Dont forget the -h if you need a help and instant documentation
nnUNet_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -t TASK_NAME_OR_ID -m CONFIGURATION --save_npz

```
If you wish to run ensembling, you can ensemble the predictions from several configurations with the following command:
```bash
#You can specify an arbitrary number of folders, but remember that each folder needs to contain npz files that were generated by nnUNet_predict
#You can also specify a file that tells the command how to postprocess.
#These files are created when running nnUNet_find_best_configuration
#(RESULTS_FOLDER/nnUNet/CONFIGURATION/TaskXXX_MYTASK/TRAINER_CLASS_NAME__PLANS_FILE_IDENTIFIER/postprocessing.json or RESULTS_FOLDER/nnUNet/ensembles/TaskXXX_MYTASK/ensemble_X__Y__Z--X__Y__Z/postprocessing.json). You can also choose to not provide a file (simply omit -pp) and nnU-Net will not run postprocessing.
#Note that per default, inference will be done with all available folds. We very strongly recommend you use all 5 folds. Thus, all 5 folds must have been trained prior to running inference. The list of available folds nnU-Net found will be printed at the start of the inference.

nnUNet_ensemble -f FOLDER1 FOLDER2 ... -o OUTPUT_FOLDER -pp POSTPROCESSING_FILE


```

## How can I use my local data to try to improve the pre-trained model
This framework allows us to load an already trained model and to train it again in order to refine the results, provided that the new database we are going to call the local database is labeled to use it for training as well as it respects the properties of the database, you can consult them by using the following command
```bash
nnUNet_print_pretrained_model_info Task001_BrainTumour
```
If we want to make a summary of the method to follow:

- Add the new data in the right directory in (nnUNet_raw_data_base/nnUNet_raw_data/TaskXXX_MYTASK, also see [here](dataset_conversion.md#How-to-update-an-existing-dataset)).

- Restart the preprocessing pipe line to verify the dataset integrity.
```bash
nnUNet_plan_and_preprocess -t XXX --verify_dataset_integrity
```
- Load the parameters of the pre-trained model if you want to use an existing model and restart the training using the command bellow:
```bash
nnUNet_train -h # To see documentation and all possible options
#use the option -pretrained_weights and find the path to the pretrained model
#Exemple hwo to use this option to train the model of the TAS01 (brain tumor segmentation
#For FOLD in [0, 1, 2, 3, 4], run:
#CONFIGURATIONS [2d , 3d_fullres , 3d_lowres , 3d_cascade_fullres ]
#TRAINER_CLASS_NAME [nnUNetTrainerV2 , nnUNetTrainerV2CascadeFullRes ]
#Dont forget to dapt the path to the .model file 
nnUNet_train CONFIGURATION TRAINER_CLASS_NAME Task001_BrainTumour FOLD -pretrained_weights nnUNet/nnUNet_trained_models/nnUNet/3d_fullres/Task001_BrainTumour/nnUNetTrainerV2__nnUNetPlansv2.1/fold_FLOD/model_final_checkpoint.model -val --npz

```

If you want to use a new database with other modalities and that does not respect the same properties of the database used for the training you will have to create a new task that you find an example how to do it  [here](dataset_conversion.md).

You can also modify or hear the preprocessing pipeline with other processing, you find more details [here](extendingnnUnet.md) .

Once the model has finished learning you can use it for segmentation in the same way as [here](#How-to-use-the-pre-trained-model-to-segment-brain-tumors).

## How can I prepare my data
Before using a new database you should put it in the right format as shown in this [example](data_format_inference.md), and check its integrity with the following command:
```bash
nnUNet_plan_and_preprocess -t XXX --verify_dataset_integrity
```
Once you have verified these two steps you can launch the training of your model as was done [here](nnU-Net.md#model-training) .

Once the training is finished you can use the model either for a prediction with the best architecture (example [here](nnU-Net.md#run-inference) or [here](inference_example_Prostate.md)) or assemble several predictions as here.

It is also possible to extend or modify some parts to adapt them to your needs