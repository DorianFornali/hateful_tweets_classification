# Universitary project in the first year of Master at Université Cote d'Azur

The goal was to implement some machine learning models that could do classification on tweets, to detect potentially hateful ones.
This is done using a dataset extracted from https://github.com/t-davidson/hate-speech-and-offensive-language

## USAGE

From the root repository <br>
```bash
python ./src/main.py [options]
Options:
-v: verbose
-t: training | Trains the models on the data prepared by train_test_splitting.py
-e: embedding | Embeds the tweets using word embeddings, NOT WORKING
-i: interactive | Allows the user to enter tweets and get predictions
Example: python main.py -v -i
```

The models are trained on bag of words.<br>
By default, we do not use oversampling however you can decide to switch it on by setting the OVERSAMPLING variable in train_test_splitting.py to True.
