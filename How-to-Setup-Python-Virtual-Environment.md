#How to Setup Python Virtual Environment

You have the choice to use **Conda** or **Pip**.

[ Conda ](#Setup Conda) creates language-agnostic environments natively whereas [ pip ](#Setup PIP) relies on virtualenv to manage only Python environments Though it is recommended to always use conda packages, conda also includes pip, so you don't have to choose between the two.

#Setup Conda
## 1) Check if conda is installed in your path.
- Open up the anaconda command prompt.
- Type conda -V  and press enter.
- If the conda is successfully installed in your system you should see a similar output: ```conda 4.10.1```.
## 2) Update the conda environment 
Entre the following command:
    ```
conda update conda
    ``` 
## 3) Set up the virtual environment
+ Type ```conda search “^python$”``` to see the list of available python versions.
+ Now replace the envname with the name you want to give to your virtual environment and replace x.x with the python version you want to use.  

  ```conda create -n envname python=x.x anaconda```

Let’s create a virtual environment name nnUnet for Python3.6, we have to use this command:

    conda creat -n nnUnet python=3.6 anaconda


## 4) Activating the virtual environment
- To see the list of all the available environments use command ```conda info -e```.
- To activate the virtual environment, enter the given command and replace your given environment name with envname: ``` conda activate envname```
- For our case will be something like : ``` conda activate nnUnet```

## 5) Installation of required packages to the virtual environment
- Type the following command to install the additional packages to the environment and replace envname with the name of your environment.

    ``` conda install -n yourenvname package```

## 6) Deactivating the virtual environment
- To come out of the particular environment type the following command. The settings of the environment will remain as it is.  
```conda deactivate```

## 7) Deletion of virtual environment
- If you no longer require a virtual environment. Delete it using the following command and replace your environment name with envname  
```conda remove -n envname -all```


#Setup PIP
