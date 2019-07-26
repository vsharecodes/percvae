## perCVAE Public

### 1.Environments
```
TensorFlow 1.4.0
cuDNN <= 7.3
Python 2.7
Numpy
NLTK
pyyaml
```
Conda virtual environment is recommended, for both python and cuda version. The following steps are based on the conda virtual environment:

First create conda virtual environment through:

```
conda create -n percvae python=2.7
```

Then install cudnn:

```
conda activate percvae

conda install cudnn
```

make sure the version of cuDNN <= 7.3.

Finally:

```
pip install -r requirements.txt
```

### 2.Dataset
Download the **Persona-Chat** dataset from **Task:ConvAI2** in [ParlAI](https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks/convai2).
More details about this dataset can be found at the ConvAI2 homepage at <http://convai.io/#personachat-convai2-dataset>.

We use the **train\_self_original\_no\_cands.txt**
and
**valid\_self\_original\_no\_cands.txt**
for training / validating and testing. For data preparation, please refer to instructions and example files in `./data`. 

### 3.How to run
####Training

```
python main.py -train 		        \
-config without_labeled_data.yaml 	\
-save_to saved_models
```
or simply:

```
python main.py -train
```

where the **.yaml** file in `./config_api` contains most parameter settings.
Training should only take a few hours (it is tested on one Tesla P100 GPU).

####Inference

```
python main.py -test 						  \
-model saved_models/model_2019_07_25_17_47_25 \
-config without_labeled_data.yaml
```

And the results will print in the terminal by default.

### 4.Examples
Our experimental code has been modified to run directly on the original Persona-Chat dataset. Here are some generated examples in our test-running with this code:

```
Persona 0: </s>
Persona 1: i used to be pretty but time has not been kind
Persona 2: i used to be a painter but now i am a housekeeper
Persona 3: i fantasize about taking over the world
Persona 4: i have two grown children who never visit
Persona 5: i am a 56 year old woman
Batch 4 index 0
Source: <s> 30 and i don ' t have any children </s>
Target >> i am 56 . what do you do for a living ?
Sample 0 >> i have two sons . i wish i could have more grandchildren as i am close .
Sample 1 >> i wish i could not have more children , i need more money .
Sample 2 >> i have two sons and a loving husband .
Sample 3 >> i am 56 . i am more of a lonely life too sick in my small town and small town .
Sample 4 >> i have two sons . i bet they do .
```

```
Persona 0: </s>
Persona 1: i enjoy being around people
Persona 2: i volunteer in a homeless shelter
Persona 3: i like to workout a a times a week
Persona 4: i am a professional wrestler
Persona 5: in my spare time i do volunteer work
Batch 31 index 0
Source: <s> hi how are you doing ? i am fine thanks to the lord . </s>
Target >> i am doing well , thanks .
Sample 0 >> i am doing well . just volunteer work
Sample 1 >> i am well , so do you work ?
Sample 2 >> i am well , i volunteer at my library .
Sample 3 >> i am well . thanks for asking . i am actually in school at work .
Sample 4 >> i love doing well ! i just volunteer work . so tell me about yourself .
```

```
Persona 0: </s>
Persona 1: i like to grill outdoors
Persona 2: i enjoy <unk> my lawn on sunny days
Persona 3: i have been retired for a years
Persona 4: i go gambling in my spare time
Batch 3205 index 0
Source: <s> i love cats and have five of them . </s>
Target >> cats are nice . how old are you ?
Sample 0 >> how many cats do you have ?
Sample 1 >> i enjoy video games . how about you ?
Sample 2 >> five does a lot of work ! i enjoy meat and baked ziti ! any advice ? restaurants ?
Sample 3 >> i love cats and all non breeds of two kids .
Sample 4 >> i have a couple nephews and a mini van .
```

* **Persona k**:  representing the persona texts assigned to each dialog session. In the dataset, each dialog session usually has 4 or 5 persona texts. **Persona 0** is **None**, which indicates no persona text is used.
* **Batch p index q**:  the order of minibatches and the index inside a minibatch.
* **Source**:  the input of the dialog, from the dataset
* **Target**:  the groundtruth response of the given input, also from the dateset
* **Sample k**:  *N* generated responses from the model. Here *N* is 5.

Finally, notice that pre-training on larger corpus such as OpenSubtitles or Twitter to get a stronger language model would almost certainly yield better results.
