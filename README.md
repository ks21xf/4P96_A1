Hi marker

Here's everything you need to know to run the code. final_pipeline.py is the file that contains the full pipeline with customizable parameters.

run this to clone:

```
git clone https://github.com/ks21xf/4P96_A1
cd 4P96_A1
```

if you want to install all the libraries:

```
pip install -r requirements.txt
```

Before running the program, make sure that your device is capable of allocating at least 12.5 GB of RAM to the program. The semisupervised portion of the pipeline requires some pretty big matrices in order to work. Also know that it takes like 20 minutes plus or minus to run the semisupervised portion. I'm sorry it has to be this way. but the rest of the network runs in a decent time window. By default the SSL portion is disabled. Below is a list of parameters to configure the network.  

| Parameter | Description | Type | Possible Values | Default Value |
| --- | ----------- |  --- | --------- | ----- |
| --lr | Adjusts the learning rate of the network | float | any |  0.01 |
| --momentum | Changes momentum of network | float | any | 0.9 |
| --norm | The normalization strategy the network uses | str | "None", "minmax" or "zscore" | "minmax" |
| --init | The way the weights are initialized in the network | str | "he", "normal", "uniform", or "None" | "normal |
| --aug | Enable data augmentation | bool | True or False | True |
| --dropout | Enable dropout (regularization) | bool | True or False | True |
| --alr | Enables an adaptive learning rate. You can pick between None, exponential decay or decay by a factor | str | "exp_decay","factor_decay" or "None" | "None" |
| --enable_ssl | Enables the semi-supervised portion of the pipeline | bool | True or False | False |

Note: please make sure for any string based parameter, you put "None" and not None!

###IF YOU AREN'T ABLE TO RUN THE CODE BECAUSE OF RAM ISSUES
I have a backup notebook file called FINAL_PIPELINE.ipynb that is the same as the py file except its just a notebook. the parameters are at the top of the first notebook. If you put the notebook in something like Colab or Kaggle then you should have enough RAM in that environment to run the code. 

Also, there are more files in there that I used during the creation of the report that aren't in the final_pipeline.py file. the one called testing_with_varying_labels.ipynb is one that I used for stage 5 to test the network with different percentages of labeled data. feel free to take a look but that file is not pretty.
