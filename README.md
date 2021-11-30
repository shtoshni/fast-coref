# On Generalization in Coreference Resolution

Code for the CRAC 2021 paper [On Generalization in Coreference Resolution](https://arxiv.org/pdf/2109.09667.pdf). 
This paper extends our work from the EMNLP 2020 paper [Learning to Ignore: Long Document Coreference
with Bounded Memory Neural Networks](https://arxiv.org/pdf/2010.02807.pdf). 


## Changelog
- Support for joint training and evaluation on eight popular coreference datasets.
- Mention proposal module filters mentions based on span-score rather than
  as a function of document length. In practice, this leads to a minor reduction in mention recall at the cost of 
  significant increase in precision i.e. less noisy mentions get filtered through.
- Inference is done in an online-fashion, one document encoder chunk 
(4096 tokens for LongFormer) at a time. Thus, this
inference can scale to very long documents.
- Support for training with pseudo-singletons.
- Switched document encoder from SpanBERT to LongFormer.
- Using Hydra for configs.

## Resources
- Pretrained models are released 
[here](https://drive.google.com/drive/folders/1270pP1JIYLleLH7rkRyXyHV2p0C7rX_8?usp=sharing). 
- Processed data for Character Identification, GAP, LitBank, QuizBowl, WikiCoref, 
and WSC is available [here](https://drive.google.com/drive/folders/1j7OsSmPhkhtuH_YvS9LvAM4fx_VoqZzw?usp=sharing).    
- The fine-tuned document encoders are separately released on 
[huggingface](https://huggingface.co/shtoshni).
- Checkout the 
[Colab Notebook](https://colab.research.google.com/drive/11ejXc1wDqzUxpgRH1nLvqEifAX30Z71_?usp=sharing)
for a demo on how to perform inference with the pretrained models.


## Environment Setup

### Install  Requirements
The codebase has been tested for:
```
python==3.8.8
torch==1.10.0
transformers==4.11.3
scipy==1.6.3
omegaconf==2.1.1
hydra-core==1.1.1
wandb==0.12.6
```
These are the core requirements which can be separately installed or just run:
```
pip install -r requirements.txt
```

Clone a few Github Repos (including this!)
```
# Clone this repo
git clone https://github.com/shtoshni/fast-coref

# Create a coref resources directory which contains the official 
# scorers and the data
mkdir coref_resources; cd coref_resources/
git clone https://github.com/conll/reference-coreference-scorers

# Create data subdirectory in the resources directory
mkdir data
```

### Data Preparation
```
cd fast-coref/src
export PYTHONPATH=.

# Demonstrating the data preparation step for QuizBowl.
# Here we point to the CoNLL directory extracted from the original data
# Output directory is created in the parent directory i.e. 
# ../../coref_resources/data/quizbowl/longformer
python data_processing/process_quizbowl.py ../../coref_resources/data/quizbowl/conll
```
- Processed data for Character Identification, GAP, LitBank, QuizBowl, WikiCoref, 
and WSC is available [here](https://drive.google.com/drive/folders/1j7OsSmPhkhtuH_YvS9LvAM4fx_VoqZzw?usp=sharing).    
- OntoNotes is a proprietary dataset and Preco is large. 
Please run the corresponding scripts to process these datasets. 
For OntoNotes we're sharing the pseudo-singletons in its namesake directory.

### Configurations
The config files are located in ``src/conf``. <br/> 
All the experiment configs are located in ``src/conf/experiment`` subdirectory. <br/>

Path strings are limited to the experiment configs and the main ``src/conf/config.yaml`` file. 
These paths can be manually edited, or overriden via command line. 

**Note**<br/> 
The default configs correspond to the configs used for training the models reported in the CRAC paper.
All models are trained for a maximum of 100K steps. 

The only exception is PreCo ([Wandb log](https://wandb.ai/shtoshni/Coreference/runs/preco_e2550d23c0a93cb5be272c3b9a484c37/overview?workspace=user-shtoshni)) where we experimented with more training steps 
(150K instead of 100K). But even there, the best validation performance is obtained 
at 60K steps and the training stops at 110K (after 10 evals without improvement). 
## Training and Inference
```
cd fast-coref/src
export PYTHONPATH=.
```

### Training
Here are a few training commands I've used.<br/>

**Joint training with wandb logging**
```
python main.py experiment=joint use_wandb=True
```

**LitBank training without label smoothing**
```
python main.py experiment=litbank trainer.label_smoothing_wt=0.0
```

**OntoNotes training with pseudo-singletons and longformer-base**
```
python main.py experiment=ontonotes_pseudo model/doc_encoder/transformer=longformer_base
```

**LitBank training with bounded memory model (memory size 20)**
```
python main.py experiment=litbank model/memory/mem_type=learned model.memory.mem_type.max_ents=20
```

**Note**<br/>
The model is saved in two parts. The document encoder and all the remaining parametes 
are saved separately. The document encoder can then be easily uploaded to 
Huggingface.

### Inference

**Inference on OntoNotes evaluation dataset with the jointly trained model**
```
python main.py experiment=ontonotes_pseudo paths.model_dir=../models/joint_best/ train=False
```

**Evaluate all the datasets + Use the huggingface uploaded document encoder**
```
python main.py experiment=eval_all paths.model_dir=../models/check_ontonotes/ model/doc_encoder/transformer=longformer_ontonotes override_encoder=True
```
The ``longformer_ontonotes`` model corresponds to the ``shtoshni/longformer_coreference_ontonotes`` 
encoder uploaded by us. 




## Miscellaneous 
### Why this repository name?
~~Marketing, Self-boasting~~ 
There are three-fold reasons for this:
- Compared to the popular mention-ranking paradigm, entity-ranking models compare 
a mention to previous entities rather than all previous mentions. 
Avereage cluster sizes in typical datasets is 3-4 mentions per cluster, implying a 3-4x 
reduction in such ranking comparisons. Note that the speedup won't be 3-4x since the 
runtime is dominated by document encoders.
- The mention proposal module in previous work (Lee et al. 2017, 18) works on a 
high-recall principle i.e. the module filters through a lot of noisy mentions which 
the clustering module has to deal with (for OntoNotes, the clustering module 
classifies them as singletons which don't account for performance evaluation). 
The model implemented in this repo uses a high-precision mention proposal module. 
Thus, the clustering module has to cluster fewer mentions which again improves speed.
- The bounded memory models are faster than their unbounded memory counterparts for 
long documents, say a few thousand words, and a small memory size, say 20. 

There are a lot of engineering hacks such as lower precision models, better implementation choices, etc., which 
can further improve the model's speed. 


### Why Hydra for configs?
It took me a few days to get a hang of Hydra but I highly recommend Hydra for 
maintaining configs. A few selling points of Hydra which made me persist:
- Clear demarkation between model config, training config, etc., 
- Compositionality as a first citizen  
- Ability to override configs from command line



### Citation
```
@inproceedings{toshniwal2021generalization,
    title = {{On Generalization in Coreference Resolution}},
    author = "Shubham Toshniwal and Patrick Xia and Sam Wiseman and Karen Livescu and Kevin Gimpel",
    booktitle = "CRAC (EMNLP)",
    year = "2021",
}

@inproceedings{toshniwal2020bounded,
    title = {{Learning to Ignore: Long Document Coreference with Bounded Memory Neural Networks}},
    author = "Shubham Toshniwal and Sam Wiseman and Allyson Ettinger and Karen Livescu and Kevin Gimpel",
    booktitle = "EMNLP",
    year = "2020",
}
```