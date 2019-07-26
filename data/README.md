## Data Preparation

In this folder, 5 data files are needed to run this code.

```
.
├── train.txt
├── valid.txt
├── test.txt
├── convai2_voacb_idf.txt
└── convai2_vocab.txt
```

### Persona-Chat Dataset
The `train.txt`, `valid.txt` and `test.txt` are example files to show the data format. Each file only contains two dialog sessions, which is not enough for model training.

* Download the Persona-Chat Dataset from [ParlAI](https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks/convai2).
* Select **train\_self_original\_no\_cands.txt**
and
**valid\_self\_original\_no\_cands.txt**.
* Partition dataset and rename as given data files.

No modification or transformation for the original data format.


### Auxiliary Data
The `convai2_voacb_idf.txt` and `convai2_vocab.txt` are ready for direct usage, no more preparation needed.

The `convai2_vocab.txt` is the **official** vocabulary list, which includes all words in the Persona-Chat dataset.

The `convai2_voacb_idf.txt` includes the tf-idf weights for all words.


### MISC
Due to the dataset license and several other issues, we will **not** release the labeled data. Some major modifications have been made to our experimental code to accommodate this situation.
