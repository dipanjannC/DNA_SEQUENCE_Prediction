# CpG Sequence Predictor

This Streamlit app predicts CpG sites in DNA sequences using a pre-trained LSTM model.

## Usage

1.  **Enter DNA sequence:** Input a DNA sequence (A, T, C, G) in the text input field.  The model is trained on sequences of length 128.  Sequences of other lengths will be handled by a different model.
2.  **Click "Predict":** The app will process the input and display the predicted output.

## Technical Details

The app uses a long short-term memory (LSTM) model trained using Optuna for hyperparameter optimization.  The model is loaded from a saved checkpoint using `joblib` and `torch`. The app handles both fixed length and variable length DNA sequences, using appropriate preprocessing for each.


## Install 

`
$ pip install streamlit torch joblib
`

## Run Streamlit App

`
$ streamlit run src/app.py
`
