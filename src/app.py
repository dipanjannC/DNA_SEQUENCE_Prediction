import streamlit as st
import torch
import joblib
import os
from pathlib import Path

from model import CpGPredictor
from data_prep_util import FixedLengthSequences,VariableLengthSequences
from predict import predict_fixed_length_DNA_sequence,predict_variable_length_DNA_sequence

project_root = Path(__file__).parents[1]

# Load the model
@st.cache_resource
def load_model():
    study = joblib.load(os.path.join(project_root, "model", 'params.pkl'))

    if not isinstance(study, dict):
        raise Exception("The loaded data is not a dictionary!")
        
    try:
        hidden_size = study["hidden_size"]
        num_layers = study["num_layers"]
        dropout = study["dropout"]
    except KeyError as e:
        raise KeyError(f"Key not found in the saved params. Details: {e}")

    model = CpGPredictor(hidden_size, num_layers, dropout)
    model.load_state_dict(torch.load(os.path.join(project_root, "model", "CpG_predictor.pth"), map_location=torch.device("cpu")))
    model.eval()
    return model

# Prepare tensor
model = load_model()

# Streamlit UI
st.title("CpG Sequence Predictor")
st.write("This app predicts CpG sequences using a trained LSTM model.")



# User input
input_sequence = st.text_input("Enter DNA sequence of fixed length")


if input_sequence:
    if len(input_sequence) != 128:
            st.info("Input sequence not with 128 values.")
            input_int_sequence = VariableLengthSequences().dnaseq_to_intseq(sequence=input_sequence)
    else:
        st.info("Input sequence with 128 values.")
        input_int_sequence = FixedLengthSequences().dnaseq_to_intseq(sequence=input_sequence)
        


# Process input and make predictions
if st.button("Predict"):
    try:
        if len(input_int_sequence) == 128:
            pred_result = predict_fixed_length_DNA_sequence(input_int_sequence,model=model)
        else:
            pred_result = predict_variable_length_DNA_sequence(input_int_sequence,model=model)
        
        print("Prediction: ",pred_result)

        if pred_result is not None:
            st.success(f"Predicted Output: {pred_result}")
        else:
            st.error(f"Unable to predict")

    except Exception as e:
        st.error(f"Details:{e}")
