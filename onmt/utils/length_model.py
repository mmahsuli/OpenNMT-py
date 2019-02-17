import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.utils.rnn
import time

import configargparse
import onmt
import onmt.opts as opts
import onmt.model_builder
from onmt.utils.logging import init_logger


class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab, device):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(len(vocab.itos), embedding_dim).to(device)
        self.vocab = vocab
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, bidirectional=True, num_layers=2)

        # The linear layer that maps from hidden state space to the 1D score space
        self.hidden2score1 = nn.Linear(hidden_dim, 30)
        self.hidden2score2 = nn.Linear(30, 1)

    def init_hidden(self, batch_size, device):
        # Before we've done anything, we don't have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (autograd.Variable(torch.zeros(4, batch_size, self.hidden_dim // 2)).to(device),
                    autograd.Variable(torch.zeros(4, batch_size, self.hidden_dim // 2)).to(device))

    def forward(self, sentence, vocab, device):
        embeds = self.word_embeddings(sentence)
        embeds = torch.transpose(embeds, 0, 1)
        self.hidden = self.init_hidden(len(sentence), device)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        pads = (sentence == vocab.stoi['<blank>']).view(-1)
        sl, bs, _ = lstm_out.size()
        lstm_out = lstm_out.view(-1, self.hidden_dim).clone()
        lstm_out[pads, :] = -100000000
        lstm_out = lstm_out.view(sl, bs, self.hidden_dim)
        lstm_out = lstm_out.max(0)[0]
        score_space = self.hidden2score1(lstm_out.view(-1, self.hidden_dim))
        score_space = self.hidden2score2(score_space)
        return torch.transpose(score_space.view(-1, len(sentence)), 0, 1)


class MyCollator(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, data):
        def padded_sequence(sequences, pad_value):
            max_length = max(len(s) for s in sequences)
            return [
                s + ([pad_value] * (max_length - len(s)))
                for s in sequences
            ]

        # there is no need to use cuda() for words, since prepare_sequence() takes care of it
        sentences = [
                    # ([word_to_ix['BOS']] + i + [word_to_ix['EOS']])
            (i)
                    for (i,j) in data
        ]

        length_ratios = ([[j] for (i,j) in data])

        padded_words = padded_sequence(sentences, self.vocab.stoi['<blank>'])
        padded_length_ratios = padded_sequence(length_ratios, -1)

        return torch.cat([torch.LongTensor(s).unsqueeze(0) for s in padded_words]), \
               torch.cat([torch.FloatTensor(s).unsqueeze(0) for s in padded_length_ratios])


def prepare_sequence(seq, vocab):
    idxs = [(vocab.stoi[w] if w in vocab.stoi else vocab.stoi['<unk>']) for w in seq]
    return idxs


def loss_function(output, target):
    target = (target.view(-1))
    output = output.contiguous().view(-1)
    diffs = (output - target) ** 2
    return diffs.mean()


def eval_unpadded_loss(data, target, model, vocab, device):
    data, target = data.to(device), target.to(device)
    with torch.no_grad():
        data, target = autograd.Variable(data), autograd.Variable(target)
        output = model(data, vocab, device)
        return output, loss_function(output, target)


def train(opt, vocab):
    # record starting time of the program
    start_time = time.time()

    torch.manual_seed(1)

    # load training data
    train_src_loc = opt.train_src
    train_tgt_loc = opt.train_tgt
    valid_src_loc = opt.valid_src
    valid_tgt_loc = opt.valid_tgt
    EMBEDDING_DIM = opt.embedding_dim
    HIDDEN_DIM = opt.hidden_dim
    BATCH_SIZE = opt.batch_size
    EPOCHS = opt.epochs
    device = opt.device
    train_from = opt.train_from
    LOAD_MODEL = len(train_from) > 0
    train_data_limit = opt.train_data_limit
    valid_data_limit = opt.valid_data_limit
    save_model_loc = opt.save_model
    save_checkpoint_epochs = opt.save_checkpoint_epochs

    training_data = []

    target_scale = 1

    # word_to_ix = {}
    # word_to_ix['PAD'] = 0
    # word_to_ix['UNK'] = 1
    # word_to_ix['BOS'] = 2
    # word_to_ix['EOS'] = 3

    if train_data_limit > 0:
        print('Limited the training data to first {} samples'.format(train_data_limit))
    else:
        print('Using the full training dataset.')
    print('Loading the training dataset...')

    with open(train_src_loc, 'r') as train_src_file, open(train_tgt_loc, 'r') as train_tgt_file:
        for train_src_line, train_tgt_line in zip(train_src_file, train_tgt_file):
            sent, ratio = (train_src_line.strip().split(), float(train_tgt_line))
            # for word in sent:
            #     if word not in word_to_ix:
            #         word_to_ix[word] = len(word_to_ix)
            sent = prepare_sequence(sent, vocab)
            training_data.append((sent, ratio))
            train_data_limit -= 1
            if train_data_limit == 0:
                break
    print('Successfully loaded the training dataset.')
    print("EMBEDDING_DIM = {}\nHIDDEN_DIM = {}\nBATCH_SIZE = {}\nEPOCHS = {}"
          .format(EMBEDDING_DIM, HIDDEN_DIM, BATCH_SIZE, EPOCHS))

    # Train the model:
    model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, vocab, device).to(device)

    if LOAD_MODEL:
        model.load_state_dict(torch.load(train_from))
        print('Resuming the model from {0}'.format(train_from))
        model.train()

    optimizer = optim.Adam(model.parameters())

    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print("\t", param_tensor, "\t", model.state_dict()[param_tensor].size())

    # Print optimizer's state_dict
    print("Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        print("\t", var_name, "\t", optimizer.state_dict()[var_name])

    valid_data = []
    if valid_data_limit > 0:
        print('Limited the test data to first {} samples'.format(valid_data_limit))
    else:
        print('Using the full validation dataset.')
    print('Loading the validation dataset...')
    with open(valid_src_loc, 'r') as valid_src_file, open(valid_tgt_loc, 'r') as valid_tgt_file:
        for valid_src_line, valid_tgt_line in zip(valid_src_file, valid_tgt_file):
            sent, tag = (valid_src_line.strip().split(), float(valid_tgt_line))
            sent = prepare_sequence(sent, vocab)
            valid_data.append((sent, tag))
            valid_data_limit -= 1
            if valid_data_limit == 0:
                break
    print('Successfully loaded the validation dataset.')

    my_collator = MyCollator(vocab)
    # collate also does the normalization of lengths
    valid_loader = torch.utils.data.DataLoader(
        valid_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        collate_fn=my_collator
    )

    # calculate total MSE loss on the test dataset before training the model
    initial_total_loss = 0
    for batch_idx, (data, target) in enumerate(valid_loader):
        _, loss = eval_unpadded_loss(data, target, model, vocab, device)
        initial_total_loss += loss
    initial_total_loss /= valid_data.__len__()
    print('Total MSE loss before training: {}'.format(initial_total_loss))

    for epoch in range(EPOCHS):

        print("Starting epoch {}/{}...".format(epoch+1, EPOCHS))
        # this is a batch
        # collate also does the normalization of lengths
        train_loader = torch.utils.data.DataLoader(
            training_data,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=8,
            collate_fn=my_collator
        )

        for batch_idx, (data, target) in enumerate(train_loader):
            # print('Train batch id: {}/{}'.format(batch_idx, train_loader.__len__()))
            data, target = data.to(device), target.to(device)
            # Get our inputs ready for the network (turn them into Variables of word indices)
            data, target = autograd.Variable(data), autograd.Variable(target)

            # Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Also, we need to clear out the hidden state of the LSTM,
            # detaching it from its history on the last instance.
            # model.hidden = model.init_hidden()
            # ? why the line above is commented?

            # Run our forward pass.
            output = model(data, vocab, device)
            loss = loss_function(output, target)
            loss.backward()
            optimizer.step()
            # if divmod(batch_idx, 100)[1] == 0:
            #     gc.collect()
            #     print('Collected garbage.')
        # calculate total MSE loss on the test dataset after training the model on each epoch
        total_loss = 0
        for batch_idx, (data, target) in enumerate(valid_loader):
            _, loss = eval_unpadded_loss(data, target, model, vocab, device)
            total_loss += loss
        total_loss /= valid_data.__len__()
        print('Total MSE loss after training on epoch {}: {}'.format(epoch+1, total_loss))
        if divmod(epoch+1, save_checkpoint_epochs)[1] == 0:
            save_mode_on_epoch_n = save_model_loc+'_step_'+repr(epoch+1)+'.pt'
            torch.save(model.state_dict(), save_mode_on_epoch_n)
            print('Saved the model to {0}'.format(save_mode_on_epoch_n))

    print('Training completed!')
    # show the time consumed by the program
    print("Total run time: {}".format(str(time.time()-start_time)))


#TODO
def test(opt, vocab):
    # load training data
    test_src_loc = opt.test_src
    test_tgt_loc = opt.test_tgt
    EMBEDDING_DIM = opt.embedding_dim
    HIDDEN_DIM = opt.hidden_dim
    BATCH_SIZE = opt.batch_size
    EPOCHS = opt.epochs
    device = opt.device
    test_data_limit = opt.test_data_limit
    model_loc = opt.model
    output_loc = opt.output

    # Construct the model:
    model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, vocab, device).to(device)

    model.load_state_dict(torch.load(model_loc))
    model.eval()


    # print("Model's state_dict:")
    # for param_tensor in model.state_dict():
    #     print("\t", param_tensor, "\t", model.state_dict()[param_tensor].size())


    test_data = []
    if test_data_limit > 0:
        print('Limited the test data to first {} samples'.format(test_data_limit))
    else:
        print('Using the full test dataset.')
    print('Loading the test dataset...')
    with open(test_src_loc, 'r') as valid_src_file, open(test_tgt_loc, 'r') as valid_tgt_file:
        for test_src_line, test_tgt_line in zip(valid_src_file, valid_tgt_file):
            sent, tag = (test_src_line.strip().split(), float(test_tgt_line))
            sent = prepare_sequence(sent, vocab)
            test_data.append((sent, tag))
            test_data_limit -= 1
            if test_data_limit == 0:
                break
    print('Successfully loaded the test dataset.')

    my_collator = MyCollator(vocab)
    # collate also does the normalization of lengths
    valid_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=8,
        collate_fn=my_collator
    )

    output_file = open(output_loc, '+w')
    total_loss = 0
    for batch_idx, (data, target) in enumerate(valid_loader):
        output, loss = eval_unpadded_loss(data, target, model, vocab, device)
        output_file.write(output)
        total_loss += loss
    total_loss /= test_data.__len__()
    print('Total MSE loss: {}'.format(total_loss))

