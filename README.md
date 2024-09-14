### Character-Level Language Model for Pet Names

This is a character-level language model trained on a pet names dataset. Given the first three characters of a word, the model predicts the next character in the sequence. The goal of this project is to generate new pet names that fit within the same style as the original dataset. This project is inspired by the Andrej Karpathy's [[MAKEMORE]](https://github.com/karpathy/makemore/) project

### Dataset
The dataset used for training this model is a collection of almost 14k unique pet names. The dataset file is included in the repository as names.txt.

### Model Architecture
The model architecture is based on MLP layers along with BN layers with the tanh as an activation function. The model is trained using batches of sequences, where each sequence is made up of the first three characters of a pet name and the next character in the sequence. The output of the model is a probability distribution over the set of possible next characters.

The model is implemented from scratch. 

### Training
To train the model, run the ```charflow.py``` script using the following command line:

```
python charflow.py -f file_path -s steps -e embedding_size -hu hidden_units_size -b batch_size -c context_size
```

Default values:
- steps: 200000
- embedding_size: 20
- hiden_units_size: 100
- batch_size: 32
- context_size: 3

### Results
Here are some examples of pet names generated by the model:
```
RAZE
CALIE
DOO
KIT
SAMSES
MACGUINNY
NAGIYA
SIDNEST
KIBBLES
BARNOLD
```
As you can see, the generated pet names are consistent with the style of the original dataset. However, some of the names may not be entirely realistic or usable.

### Conclusion
This project demonstrates the use of character-level language models for generating new text based on a given input. While this particular model was trained on pet names, the same approach could be applied to other domains and datasets.

### License
MIT
