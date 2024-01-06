### Trensformers for language modeling

This code is for language modeling with transformers.
Text classification task and masked language modeling has been implemented.


-  `./src/base.py`: self attnetion and transformer block
-  `./src/transformer.py`: network architecture for text classification
-  `./experiments/classify.py`: training text classifier network 
-  `./experiments/masked_language_modeling`: Hugging face-based model for masked language modeling  

### how to run the models

first install the package. 

important: when install the package make sure the current directory is where `setup.py` file exists. 

```python
pip install -e .
```
`e` means the package is editable.

Then change the current directory to the directory that the model is.

