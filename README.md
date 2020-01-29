# FastMAML
Lightweight but powerful PyTorch 1.3 implementation of MAML (model-agnostic meta learning) from the 2017 paper by Finn et al. Inspired in part by the original implementation in TensorFlow 1.x and by Kate Rakelly's PyTorch 0.x implementation.

## Dataset

This model uses the Omniglot dataset by default. To download it, go to https://github.com/tristandeleu/ntm-one-shot/tree/master/data/omniglot and follow the instructions. By default, this implementation assumes a 28x28 size, but this can be changed.

## Running 

This implementation comes with a variety of configurable parameters; these can be changed in main.py. To run with the default settings, just install the requirements in `requirements.txt` and run `python main.py`.

## Output

Output is saved by default to the `./output` directory, which will be created if none exists. For a given training run, the model's state is saved every 10 epochs but only the last three saves are retained. The model tracks the performance on the meta-training set at every interval, in addition to the meta-validation set, and saves these results as `.npy` files in the `meta_train` and `meta_validation` folders. 

## Results

Meta-training results on the Omniglot dataset over 5,000 iterations, smoothed over an interval of 10:


![Alt text](./graphs/meta_validation_query_accuracy.png?raw=true "Title")