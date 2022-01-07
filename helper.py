import os
import torch


class Vocabulary:
    def __init__(self, pad_token="<pad>", unk_token='<unk>'):

        # add the default pad token and the default unknown token
        self.id_to_string = {0: pad_token, 1: unk_token}
        self.string_to_id = {pad_token: 0, unk_token: 1}

        # shortcut access
        self.pad_id = 0
        self.unk_id = 1

    def __len__(self):
        return len(self.id_to_string)

    def add_new_char(self, string):
        self.string_to_id[string] = len(self.string_to_id)
        self.id_to_string[len(self.id_to_string)] = string

    # Given a string, return ID
    def get_idx(self, string, extend_vocab=False):
        if string in self.string_to_id:
            return self.string_to_id[string]
        elif extend_vocab:  # add the new char
            self.add_new_char(string)
            return self.string_to_id[string]
        else:
            return self.unk_id


# Use vocab to map the characters in input
def map_to_IDs(input, vocab):
    output = []
    for token in input:
        output.append(vocab.get_idx(token))
    return output


# Use vocab to map from the IDs in input to the corresponding character
def map_from_IDs(input, vocab):
    output = []
    for token in input:
        output.append(vocab.id_to_string[token])
    return output


# Read the raw txt file and generate a 1D PyTorch tensor
# containing the whole text mapped to sequence of token IDs, and a vocab object.
class LongTextData:

    def __init__(self, file_path, vocab=None, extend_vocab=True, device='cuda'):
        # extend_vocab: bool, if True extend the vocab
        self.extend_vocab, self.device = extend_vocab, device
        self.data, self.vocab = self.text_to_data(file_path, vocab)

    def __len__(self):
        return len(self.data)

    def text_to_data(self, text_file, vocab):
        """Read a raw text file and create its tensor and the vocab.

        Args:
          text_file: a path to a raw text file.
          vocab: a Vocab object

        Returns:
          Tensor representing the input text, vocab file

        """
        assert os.path.exists(text_file)
        if vocab is None:
            vocab = Vocabulary()

        # Construct data
        full_text = []
        print(f"Reading text file from: {text_file}")
        with open(text_file, 'r', encoding="utf8") as text:
            for line in text:
                tokens = list(line)
                for token in tokens:
                    # get index will extend the vocab if the input
                    # token is not yet part of the text.
                    full_text.append(vocab.get_idx(token, extend_vocab=self.extend_vocab))

        # convert to tensor
        data = torch.tensor(full_text, device=self.device, dtype=torch.int64)
        print("Done.")

        return data, vocab


# Since there is no need for schuffling the data, we just have to split
# the text data according to the batch size and bptt length.
# The input to be fed to the model will be batch[:-1]
# The target to be used for the loss will be batch[1:]
class ChunkedTextData:

    def __init__(self, data, bsz, bptt_len, pad_id):
        # bsz: int, batch size
        # bptt_len: int, bptt length
        # pad_id: int, ID of the padding token
        self.bsz, self.bptt_len, self.pad_id = bsz, bptt_len, pad_id
        self.batches = self.create_batch(data)

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        return self.batches[idx]

    def create_batch(self, input_data):
        """Create batches from a TextData object .

        Args:
          input_data: a TextData object.

        Returns:
          List of tensors representing batches

        """
        batches = []  # each element in `batches` is (len, B) tensor
        text_len = len(input_data)
        segment_len = text_len // self.bsz + 1

        padded = input_data.data.new_full((segment_len * self.bsz,), self.pad_id)
        padded[:text_len] = input_data.data
        padded = padded.view(self.bsz, segment_len).t()
        num_batches = segment_len // self.bptt_len + 1

        for i in range(num_batches):
            # Prepare batches such that the last symbol of the current batch
            # is the first symbol of the next batch.
            if i == 0:
                # Append a dummy start symbol using pad token
                batch = torch.cat(
                    [padded.new_full((1, self.bsz), self.pad_id),
                     padded[i * self.bptt_len:(i + 1) * self.bptt_len]], dim=0)
                batches.append(batch)
            else:
                batches.append(padded[i * self.bptt_len - 1:(i + 1) * self.bptt_len])

        return batches
