import torch
from torch.nn.utils.rnn import pad_packed_sequence,pack_padded_sequence,pad_sequence

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, lists, labels) -> None:
        self.lists = lists
        self.labels = labels

    def __getitem__(self, index):
        return torch.LongTensor(self.lists[index]), self.labels[index]

    def __len__(self):
        return len(self.lists)

    
# this will be a collate_fn for dataloader to pad sequence  
class PadSequence:
    def __init__(self, padding_value=1000):
        self.padding_value = padding_value

        
    def __call__(self, batch,target_seq_len=128):
        
        # Separate sequences and labels
        sequences, labels = zip(*batch)

       # Truncate or pad sequences to the target sequence length
        processed_sequences = []
        sequence_lengths = []
        for seq in sequences:
            if len(seq) > target_seq_len:  # Truncate
                processed_sequences.append(seq[:target_seq_len])
                sequence_lengths.append(target_seq_len)
            else:  # Pad
                # Pad sequences to the target sequence length
                processed_sequences.append(seq)
                sequence_lengths.append(len(seq))

        # Pad sequences to the maximum length in the batch (target_seq_len)
        # Input Shape : batch_size,List of tensors with varying lengths
        
        padded_sequences = pad_sequence(
            processed_sequences, batch_first=True, padding_value=self.padding_value , padding_side="right"
        )

        # Pack the padded sequences for RNN processing
        # Shape: (batch_size, target_seq_len)
        packed_sequences = pack_padded_sequence(
            padded_sequences, sequence_lengths, batch_first=True, enforce_sorted=False
        )

        # Unpack the packed sequences back to padded form
        # Shape: (batch_size, target_seq_len)
        unpacked_sequences, _ = pad_packed_sequence(
            packed_sequences, batch_first=True, total_length=target_seq_len
        )

        # Convert labels to tensor
        unpacked_sequences = unpacked_sequences.to(torch.float32)
        labels = torch.tensor(labels, dtype=torch.float32)

        return unpacked_sequences, labels