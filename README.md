# N-Gram Language Model

## Overview
This Python script implements a language model using n-grams. The script provides functionality for training the model on text data, scoring sequences of tokens, calculating perplexity, and generating sentences based on the trained model.

## Requirements
- Python 3.x
- NumPy

## Usage
1. **Training the Model**: To train the language model, provide a text file containing training data. Update the file path in the script to point to your training file. The script will tokenize the data and train the model.

2. **Testing**: Testing the model involves providing a separate text file for evaluation. Again, update the file path in the script to point to your testing file. The script will tokenize the test data and evaluate the model's performance by scoring and calculating perplexity.

3. **Generating Sentences**: After training the model, you can generate sentences using the `generate()` method. Specify the number of sentences you want to generate, and the script will produce them based on the trained model.

## Files
- `language_model.py`: The main Python script containing the implementation of the language model.
- `training_files/berp-training.txt`: Sample training data file. Update with your own training data.
- `testing_files/berp-test.txt`: Sample testing data file. Update with your own testing data.

## How to Run
1. Clone or download the repository to your local machine.
2. Ensure Python 3.x and NumPy are installed.
3. Update the file paths in the script to point to your training and testing data.
4. Run the script using `python language_model.py`.

## Additional Notes
- The script provides options for adjusting n-gram order and tokenization method (character-level or word-level).
- Performance can be further optimized, especially for large datasets, by implementing more efficient algorithms for n-gram creation and probability calculation.

## Credits

The data used in this project was obtained from [here](https://www1.icsi.berkeley.edu/Speech/berp.html). I would like to express my gratitude to the creators of the "BeRP" dataset for making it available for research and development purposes.

