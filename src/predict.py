import torch
from typing import List
from utils import MyDataset, PadSequence
from torch.utils.data import DataLoader


def predict_fixed_length_DNA_sequence(input_sequence:List[int],model,device="cpu"):
    """
    Predict CpG counts for fixed length DNA Sequence.
    
    Args:
        sequence: A single sequence (list of integers).
        model: Trained model.
        device: Device to run the model on ("cpu" or "cuda").
    Returns:
        The model's prediction (rounded to the nearest integer).
    """
    try:
        # Convert input to a tensor
        input_tensor = torch.tensor(input_sequence, dtype=torch.float32).unsqueeze(0).to(device)

        #Set model to evaluation mode
        model.eval()

        # Make prediction
        with torch.no_grad():
            prediction = model(input_tensor).item()
            prediction = torch.round(prediction)
            print("prediction")
        
        return int(prediction)
        
    except Exception as e:
        raise Exception(f'Model was not able to predict.\n Details: {e}')


def predict_variable_length_DNA_sequence(input_sequence: List[int], model, device="cpu"):
    """
    Predict CpG counts for variable fixed length DNA Sequence.
    
    Args:
        sequence: A single sequence (list of integers).
        model: Trained model.
        device: Device to run the model on ("cpu" or "cuda").
    Returns:
        The model's prediction (rounded to the nearest integer).
    """
    try:
        # Create dataset with single sequence
        test_dataset = MyDataset([input_sequence], [0])  # Single sequence with dummy label
        
        # Create dataloader with padding
        test_loader = DataLoader(
            test_dataset, 
            batch_size=1, 
            shuffle=False,
            collate_fn=PadSequence(padding_value=1000)
        )
        
        # Set model to evaluation mode
        model.eval()
        
        # Get prediction
        with torch.no_grad():
            batch, _ = next(iter(test_loader))
            batch = batch.to(device)
            output = model(batch)
            prediction = torch.round(output)
        
        return int(prediction)
    
    except Exception as e:
        raise Exception(f'Model was not able to predict.\n Details: {e}')