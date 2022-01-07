import os.path

import numpy as np
import torch
import helper
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

text_path = os.path.join("data", "49010-0.txt")
text_path2 = os.path.join("data", "TvSeries.txt")


# Trainer
class Trainer:
    def __init__(self, model, loss_fn, optimizer):
        self.model = model.to(DEVICE)
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def train(self, num_epochs, batches, prompt=None, vocab=None):
        batch_size = batches[0].size(1)
        self.model.train()
        losses = []
        perplexity = []
        start_time = time.time()
        for epoch in range(num_epochs):
            state = self.model.init_state(batch_size)
            loss_total = 0
            for batch in tqdm(batches):
                self.optimizer.zero_grad()
                input = batch[:-1]
                target = batch[1:]
                target = target.flatten()
                output, state = self.model(input, state)

                loss = self.loss_fn(output, target)
                self.optimizer.zero_grad()  # Reset gradients
                loss.backward()  # Compute gradients
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # Clipping threshold == 1.0
                self.optimizer.step()  # Update parameters
                loss_total += loss.item()
            losses.append(loss_total / len(batches))  # Compute the mean between all the losses in each batch
            perplexity.append(np.exp(losses[-1]))
            print(f'epoch: {epoch}, train loss: {losses[-1]}, perplexity: {perplexity[-1]}')
            with torch.no_grad():
                if prompt is not None and vocab is not None:
                    greedy_prediction = predict(self.model, vocab, prompt, 100)
                    sampling_prediction = predict(self.model, vocab, prompt, 100, False)
                    print(f'prompt: {prompt}')
                    print(f'greedy prediction: {greedy_prediction}')
                    print(f'sampling prediction: {sampling_prediction}')
            if perplexity[-1] < 1.03:
                minutes_elapsed = round((time.time() - start_time) / 60, 4)
                print(f"Perplexity of the model: {perplexity[-1]}. Model well trained after {minutes_elapsed} minutes.")
                break

        return self.model


# Two decoding algorithms: greedy (greedy == True) and sampling (greedy == False)
class LSTM(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layers, vocabulary_size, pad_id, cross_entropy=True):
        super(LSTM, self).__init__()

        self.cross_entropy = cross_entropy
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocabulary_size, embedding_size, padding_idx=pad_id)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocabulary_size)
        self.softmax = nn.Softmax(dim=1)

    def greedy_decoding(self, x, prev_state):
        output, state = self.forward(x, prev_state)
        output = self.softmax(output)
        new_output = torch.topk(output, k=1, dim=-1).indices
        return new_output, state

    def sampling_decoding(self, x, prev_state):
        output, state = self.forward(x, prev_state)
        output = self.softmax(output)
        new_output = torch.multinomial(output, num_samples=1)
        return new_output, state

    def forward(self, x, prev_state):
        embeds = self.embedding(x)
        output, state = self.lstm(embeds, prev_state)
        output = output.view(-1, self.hidden_size)
        output = self.linear(output)
        if not self.cross_entropy:
            output = self.softmax(output)
        return output, (state[0].detach(), state[1].detach())

    def init_state(self, batch_size):
        init_state = (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(DEVICE),
                      torch.zeros(self.num_layers, batch_size, self.hidden_size).to(DEVICE))
        return init_state


def text_data_properties():
    my_data = helper.LongTextData(text_path, device=DEVICE)
    txt = open(text_path, 'r', encoding="utf8").read()
    paragraphs_n = len(txt.split("\n\n"))
    lines = txt.split("\n")
    words = txt.split(" ")
    lines_n = len(lines)
    words_n = len(words)
    average_lines_length = sum(map(len, lines)) / len(lines)
    average_words_length = sum(map(len, words)) / len(words)
    upp_character = sum([c.isupper() for c in txt])
    low_character = sum([c.islower() for c in txt])
    print(f"Size of character vocabulary: {len(my_data.vocab)}.")
    print(f"Number of characters in the file: {len(txt)}.")
    print(f"Number of words: {words_n}.")
    print(f"Number of paragraphs: {paragraphs_n}.")
    print(f"Number of lines: {lines_n}.")
    print(f"Average line length: {average_lines_length:4.2f}.")
    print(f'Average word length: {average_words_length:4.2f}.')
    print(f"Percentage of capitalized letters: {(upp_character * 100 / (upp_character + low_character)):4.2f}.")


def run_model(lr=0.001, bptt_len=64):
    batch_size = 32
    num_epochs = 300  # High since likely the training will stop when the perplexity will be below 1.03
    learning_rate = lr
    pad_id = 0

    my_data = helper.LongTextData(text_path, device=DEVICE)
    batches = helper.ChunkedTextData(my_data, batch_size, bptt_len, pad_id=pad_id)
    prompt = "A KID coming home "

    hidden_size = 2048
    num_layers = 1
    embedding_size = 64
    vocabulary_size = len(my_data.vocab)

    model = LSTM(embedding_size, hidden_size, num_layers, vocabulary_size, pad_id)

    # Create loss and optimizer
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)  # Ignore pad_id
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    trainer = Trainer(model, loss_fn, optimizer)
    model_trained = trainer.train(num_epochs, batches.batches, prompt=prompt, vocab=my_data.vocab)

    return model_trained, my_data.vocab


def run_model2(lr=0.001, bptt_len=64):
    batch_size = 32
    num_epochs = 300  # High since likely the training will stop when the perplexity will be below 1.03
    learning_rate = lr
    pad_id = 0

    my_data = helper.LongTextData(text_path2, device=DEVICE)
    batches = helper.ChunkedTextData(my_data, batch_size, bptt_len, pad_id=pad_id)
    prompt = '{"id":41968,"name":"runtime exception'

    hidden_size = 2048
    num_layers = 1
    embedding_size = 64
    vocabulary_size = len(my_data.vocab)

    model = LSTM(embedding_size, hidden_size, num_layers, vocabulary_size, pad_id)

    # Create loss and optimizer
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)  # Ignore pad_id
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    trainer = Trainer(model, loss_fn, optimizer)
    model_trained = trainer.train(num_epochs, batches.batches, prompt=prompt, vocab=my_data.vocab)

    return model_trained, my_data.vocab


def predict(model, vocab, prompt, predict_len, greedy=True):
    prompt = helper.map_to_IDs(prompt, vocab)
    prediction = []
    if greedy:
        prediction_algorithm = model.greedy_decoding
    else:
        prediction_algorithm = model.sampling_decoding
    state = model.init_state(1)
    for i in range(len(prompt) - 1):
        prompt_input = torch.tensor(prompt[i], device=DEVICE, dtype=torch.int64)
        _, state = model(prompt_input.view(1)[:, None], state)
    last_char = prompt[-1]
    for _ in range(predict_len):
        last_char = torch.tensor(last_char, device=DEVICE, dtype=torch.int64)
        output, state = prediction_algorithm(last_char.view(1)[:, None], state)
        prediction.append(output.item())
        last_char = prediction[-1]
    return "".join(helper.map_from_IDs(prediction, vocab))


def main():
    text_data_properties()

    model, vocab = run_model()

    run_model(lr=0.0001)
    run_model(lr=0.01)

    run_model(bptt_len=128)
    run_model(bptt_len=32)

    print(predict(model, vocab, "A FOX AND A CRAB ", 300))
    print(predict(model, vocab, "THE CAT ON THE TABLE ", 300))
    print(predict(model, vocab, "A hawk flew high in the sky when suddenly ", 300))
    print(predict(model, vocab, "IN the year 1878 I took my degree of Doctor of Medicine of the University of London",
                  300))

    print(predict(model, vocab, "A FOX AND A CRAB ", 300, greedy=False))
    print(predict(model, vocab, "THE CAT ON THE TABLE ", 300, greedy=False))

    model2, vocab2 = run_model2()

    run_model2(lr=0.0001)
    run_model2(lr=0.01)
    run_model2(bptt_len=128)
    run_model2(bptt_len=32)

    print(predict(model2, vocab2, '{"id":1920,"name":"Covid19', 300))
    print(predict(model2, vocab2, '{"id":1920,"name":"Covid19', 300, greedy=False))


if __name__ == '__main__':
    main()
