# some config
import torch

LSTM_HIDDEN = 512
LSTM_LAYER = 2

learning_rate = 0.005
epoch_num = 40
dropout = 0.20

class CpGPredictor(torch.nn.Module):
    '''Simple model that uses an LSTM to count the number of CpGs in a sequence'''
    def __init__(self, hidden_size, num_layers,dropout):
        super(CpGPredictor, self).__init__()
        self.lstm = torch.nn.LSTM(input_size=128, hidden_size=hidden_size, num_layers=num_layers, 
                            dropout=dropout,
                            batch_first=True, bidirectional=False)
        self.fc = torch.nn.Linear(hidden_size, 1)  # Output size is 1 for regression
        self.relu = torch.nn.ReLU()

    def forward(self, x) -> torch.Tensor:
        # Initialize hidden state
        batch_size = x.size(0)
        h0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(x.device)
        
        # Reshape input to (batch_size, sequence_length, input_size)
        x = x.unsqueeze(1)
        
        # LSTM Layer
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Take output from the last time step
        last_output = lstm_out[:, -1, :]
        
        # Fully connected layer
        out = self.fc(last_output)
        out = self.relu(out)
        return out.flatten()
    
# if __name__ == __main__:
