import random
import numpy as np
import torch
from typing import Sequence, Any, List
from functools import partial

def set_seed(seed=13):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(13)

class FixedLengthSequences:
    def __init__(self, seq_len: int = 128):
        self.seq_len = seq_len
        self.sequence = 'NACGT'
        self.dna2int = {a: i for a, i in zip(self.sequence, range(5))}
        self.int2dna = {i: a for a, i in zip(self.sequence, range(5))}

    def generate_random_sequences(self, n_seqs: int) -> List[List[int]]:
        return [[random.randint(0, 4) for _ in range(self.seq_len)] for _ in range(n_seqs)]

    def intseq_to_dnaseq(self, sequence: List[int]) -> str:
        return ''.join(map(self.int2dna.get, sequence))

    def dnaseq_to_intseq(self, sequence: str) -> List[int]:
        return list(map(self.dna2int.get, sequence))



class VariableLengthSequences:
    def __init__(self, lb: int = 16, ub: int = 128):
        self.lb = lb
        self.ub = ub
        self.sequence = 'NACGT'
        self.dna2int = {a: i for a, i in zip(self.sequence, range(1, 6))}
        self.int2dna = {i: a for a, i in zip(self.sequence, range(1, 6))}
        self.dna2int.update({"pad": 1000})
        self.int2dna.update({1000: "<pad>"})

    def generate_random_sequences(self, n_seqs: int) -> List[List[int]]:
        sequences = []
        for _ in range(n_seqs):
            seq_len = random.randint(self.lb, self.ub)
            sequences.append([random.randint(1, 5) for _ in range(seq_len)])
        return sequences

    def intseq_to_dnaseq(self, sequence: List[int]) -> str:
        return ''.join(map(self.int2dna.get, sequence))

    def dnaseq_to_intseq(self, sequence: str) -> List[int]:
        return list(map(self.dna2int.get, sequence))
        


def main():
    print(FixedLengthSequences().dnaseq_to_intseq(sequence="ACGTNACGT"))
    print(VariableLengthSequences().dnaseq_to_intseq(sequence="ACGTNACGT"))



if __name__ == "__main__":
    main()

