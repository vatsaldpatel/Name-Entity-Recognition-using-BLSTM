#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np
import copy
import gzip
import sys
from sklearn.metrics import f1_score

def read_data(loc):
    f = open(loc, "r")

    # Extracting sentences form data
    lines = f.read().split("\n\n")

    # Removing the \n at the last line
    lines[-1] = lines[-1][:len(lines[-1])-1]
    return lines


def get_words_tags_mapping(lines):
    # Getting Vocabulary and tags from the Dataset
    words = []
    tags = []

    word_to_int = {}

    for line in lines:
        tmps = line.split("\n")
        for x in tmps:
            tmp = x.split()
            word_to_int[tmp[1]] = 0
            words.append(tmp[1])
            tags.append(tmp[2])

    words = list(set(words))
    tags = list(set(tags))
    
    i = 2
    for word in word_to_int.keys():
        word_to_int[word] = i
        i += 1
    word_to_int['<pad>'] = 0
    word_to_int['<unk>'] = 1
    
    tag_to_int = {'I-MISC': 1,
                 'I-PER': 2,
                 'O': 3,
                 'I-LOC': 4,
                 'B-ORG': 5,
                 'B-PER': 6,
                 'B-MISC': 7,
                 'B-LOC': 8,
                 'I-ORG': 9,
                 '<pad>': 0}
    
    return words, tags, word_to_int, tag_to_int


def map_sentences(lines, word_to_int, tag_to_int):
    sentence_word_mapping = []
    sentence_tag_mapping = []
    original_sentence_length = []
    
    for line in lines:
        word_map = []
        tag_map = []
        tmps = line.split("\n")
        for x in tmps:
            tmp = x.split()
            if(tmp[1] in word_to_int.keys()):
                word_map.append(word_to_int[tmp[1]])
            else:
                word_map.append(word_to_int['<unk>'])

            tag_map.append(tag_to_int[tmp[2]])

        sentence_word_mapping.append(word_map)
        sentence_tag_mapping.append(tag_map)

    # Padding Sentences to make the length of every sentence same
    longest_sentence = max(len(x) for x in sentence_word_mapping)

    for i in range(len(sentence_word_mapping)):
        original_sentence_length.append(len(sentence_word_mapping[i]))
        sentence_word_mapping[i] = sentence_word_mapping[i] + [0]*(longest_sentence - len(sentence_word_mapping[i]))
        sentence_tag_mapping[i] = sentence_tag_mapping[i] + [0]*(longest_sentence - len(sentence_tag_mapping[i]))
        
    
    return torch.Tensor(sentence_word_mapping), torch.Tensor(sentence_tag_mapping), torch.Tensor(original_sentence_length)


def test_map_sentences(lines, word_to_int):
    sentence_word_mapping = []
    original_sentence_length = []
    
    for line in lines:
        word_map = []
        tmps = line.split("\n")
        for x in tmps:
            tmp = x.split()
            if(tmp[1] in word_to_int.keys()):
                word_map.append(word_to_int[tmp[1]])
            else:
                word_map.append(word_to_int['<unk>'])

        sentence_word_mapping.append(word_map)

    # Padding Sentences to make the length of every sentence same
    longest_sentence = max(len(x) for x in sentence_word_mapping)

    for i in range(len(sentence_word_mapping)):
        original_sentence_length.append(len(sentence_word_mapping[i]))
        sentence_word_mapping[i] = sentence_word_mapping[i] + [0]*(longest_sentence - len(sentence_word_mapping[i]))
    
    return torch.Tensor(sentence_word_mapping),torch.Tensor(original_sentence_length)


"""# BiLSTM Model"""
class BiLSTM(torch.nn.Module):
    def __init__(self, embedding_size, hidden_size, total_words, linear_out_dim, embedding):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        #self.embedding = torch.nn.Embedding(num_embeddings=total_words, embedding_dim=embedding_size)
        self.embedding = torch.nn.Embedding.from_pretrained(embedding, freeze=False)
        #self.dropout = torch.nn.Dropout(p = 0.33)
        self.lstm = torch.nn.LSTM(embedding_size,hidden_size, bidirectional = True, batch_first = True, num_layers=1)
        self.dropout = torch.nn.Dropout(p = 0.33)
        self.linear = torch.nn.Linear(2*hidden_size, linear_out_dim).to(device)
        self.activation = torch.nn.ELU()
        self.classifier = torch.nn.Linear(linear_out_dim, 10).to(device)
        
    def forward(self, x, lengths, to_eval=False):
        x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x = self.embedding(x)
        x, _ = self.lstm(x)
        if(not to_eval):
            x = self.dropout(x)
        x = self.activation(self.linear(x))
        x = self.classifier(x)
        x = x.view(-1,x.shape[2])
        x = torch.nn.functional.log_softmax(x,dim=1)
        return x


def validateModel(model, dataLoader):
    lossfunction = torch.nn.CrossEntropyLoss()
    model.eval()
    prediction_list = []
    running_loss=0
    pred = []
    true = []
    with torch.no_grad():
        for i, (Xbatch, Ybatch, sentenceLen) in enumerate(dataLoader):
            Xbatch = Xbatch.to(device)
            Ybatch = torch.nn.utils.rnn.pack_padded_sequence(Ybatch, sentenceLen, batch_first=True, enforce_sorted=False)
            Ybatch, _ = torch.nn.utils.rnn.pad_packed_sequence(Ybatch, batch_first=True)
            outputs = model(Xbatch.long(), sentenceLen, True)
            pred += (torch.argmax(outputs, dim=1).float().tolist())
            true += (Ybatch.view(-1).tolist())
            loss = lossfunction(outputs.to(device), Ybatch.view(-1).type(torch.LongTensor).to(device))
            running_loss += loss.item()
    return f1_score(true, pred, average='macro'),  running_loss/len(dataLoader.dataset)


def evaluator(Model, loader):
    Model.eval()
    tag_answers = []
    with torch.no_grad():
        for inputs, labels, senlen in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = torch.nn.utils.rnn.pack_padded_sequence(labels, senlen.cpu(), batch_first=True, enforce_sorted=False)
            labels, _ = torch.nn.utils.rnn.pad_packed_sequence(labels, batch_first=True)
            outputs = Model(inputs.long(),senlen, True)
            tag_answers.append(torch.argmax(outputs.data, dim=1).tolist())
    return tag_answers


def test_evaluator(Model, loader):
    Model.eval()
    tag_answers = []
    with torch.no_grad():
        for inputs, senlen in loader:
            inputs = inputs.to(device)
            outputs = Model(inputs.long(),senlen, True)
            tag_answers.append(torch.argmax(outputs.data, dim=1).tolist())
    return tag_answers


def generate_output(lines, tag_to_int, predicted_tags):
    ans = ""
    for i in range(len(lines)):
        word_map = []
        tag_map = []
        predicted_tag_map = []
        int_to_tag = {v:k for k,v in tag_to_int.items()}
        tmps = lines[i].split("\n")
        for j in range(len(tmps)):
            tmp = tmps[j].split()
            word_map.append(tmp[1])
            tag_map.append(tmp[2])
            #print(i,tmps[j])
            predicted_tag_map.append(predicted_tags[i][j])

        s = ''
        x = 1
        for k in range(len(word_map)):
            s += str(x) + " " + word_map[k] + " " + tag_map[k] + " " + int_to_tag[predicted_tag_map[k]] + "\n"
            x += 1
        ans += s + "\n"
    return ans


def generate_test_output(lines, tag_to_int, predicted_tags):
    ans = ""
    for i in range(len(lines)):
        word_map = []
        predicted_tag_map = []
        int_to_tag = {v:k for k,v in tag_to_int.items()}
        tmps = lines[i].split("\n")
        for j in range(len(tmps)):
            tmp = tmps[j].split()
            word_map.append(tmp[1])
            #print(i,tmps[j])
            predicted_tag_map.append(predicted_tags[i][j])

        s = ''
        x = 1
        for k in range(len(word_map)):
            s += str(x) + " " + word_map[k] + " " + int_to_tag[predicted_tag_map[k]] + "\n"
            x += 1
        ans += s + "\n"
    return ans


def write_output(model, loader, loc, dev_lines, tag_to_int):
    predicted_tags = evaluator(model, loader)
    out = open(loc, "w")
    for i in generate_output(dev_lines, tag_to_int, predicted_tags)[:-1]:
        out.write(i)
    out.close()


def write_test_output(model, loader, loc, test_lines, tag_to_int):
    predicted_tags = test_evaluator(model, loader)
    out = open(loc, "w")
    for i in generate_test_output(test_lines, tag_to_int, predicted_tags)[:-1]:
        out.write(i)
    out.close()

def write_dev_output(model, loader, loc, test_lines, tag_to_int):
    predicted_tags = evaluator(model, loader)
    out = open(loc, "w")
    for i in generate_test_output(test_lines, tag_to_int, predicted_tags)[:-1]:
        out.write(i)
    out.close()

def train_bi_lstm(model, optimizer, lossfunction, scheduler, loader, dev_loader, device, n_epochs):
    valid_loss_min_lstm = np.Inf 
    bestValScore = -np.Inf
    best_lstm_model = None
    best_model_epoch_lstm = None

    for epoch in range(n_epochs):
        model.train()
        train_loss_lstm = 0.0

        for i, (train_data, target, sentenceLength) in enumerate(loader):
            train_data = train_data.to(device)
            target = target.to(device)
            target = torch.nn.utils.rnn.pack_padded_sequence(target, sentenceLength.cpu(), batch_first=True, enforce_sorted=False)
            target, _ = torch.nn.utils.rnn.pad_packed_sequence(target, batch_first=True)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(train_data.long(), sentenceLength, False)
            output = output.to(device)
            #temp = torch.argmax(output.to('cpu'), dim=1)
            loss = lossfunction(output, target.view(-1).type(torch.LongTensor).to(device))
            loss.backward()
            optimizer.step()
            train_loss_lstm += loss.item()*train_data.size(0)

        scheduler.step()
        validationScore, validationLoss = validateModel(model, dev_loader)

        if validationScore > bestValScore:
            print("Best Epoch:", epoch+1)
            bestValScore = validationScore
            bestModel = copy.deepcopy(model)


        train_loss_lstm = train_loss_lstm/len(loader.dataset)
        print("Epoch --> "+str(epoch+1)+" :", train_loss_lstm, validationScore)
    
    return bestModel


def get_glove_embeddings(loc):
    glove_embedding = {}
    with gzip.open(loc) as f:
        for line in f:
            values = line.split()
            word = values[0].decode('utf-8')
            embedding = np.asarray(values[1:], dtype='float32')
            glove_embedding[word] = embedding
    return glove_embedding


def get_glove_word_embedding(words, word_to_int):
    glove_embedding = get_glove_embeddings("glove.6B.100d.gz")
    # # Adding a new 101st Dimension to deal with Capitalization
    # glove_word_embedding = torch.zeros((len(words)+2,105))
    # count = 0
    # for word in word_to_int.keys():
    #     if word != '<pad>':
    #         if word.lower() not in glove_embedding.keys():
    #             count += 1
    #             glove_word_embedding[word_to_int[word]] = torch.Tensor(np.append(glove_embedding['unk'],[0,0,0,0,0]))
    #         else:
    #             glove_word_embedding[word_to_int[word]] = torch.Tensor(np.append(glove_embedding[word.lower()],[0,0,0,0,0]))
    #         if word[0].isupper():
    #             glove_word_embedding[word_to_int[word]][-5:] = 1
    # print(count)

    glove_word_embedding = torch.zeros((len(words)+2,100))
    count = 0
    for word in word_to_int.keys():
        if word != '<pad>':
            if word.lower() not in glove_embedding.keys():
                count += 1
                glove_word_embedding[word_to_int[word]] = torch.Tensor(glove_embedding['unk'])
            else:
                glove_word_embedding[word_to_int[word]] = torch.Tensor(glove_embedding[word.lower()])
    #print(count)
    return glove_word_embedding


def write_dev_test_on_best():
    # Preparing Training Data
    lines = read_data("data/train")
    words, tags, word_to_int, tag_to_int = get_words_tags_mapping(lines)

    best_model = BiLSTM(embedding_size = 100, hidden_size = 256, total_words = len(words)+2, linear_out_dim=128, embedding=torch.rand(len(words)+2,100)).to(device)
    best_model.load_state_dict(torch.load("model/blstm1.pt"))

    # Dev Lines
    dev_lines = read_data("data/dev")
    sentence_word_mapping, sentence_tag_mapping, original_sentence_length = map_sentences(dev_lines, word_to_int, tag_to_int)
    dev_data = torch.utils.data.TensorDataset(sentence_word_mapping, sentence_tag_mapping, original_sentence_length)
    dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=1, shuffle=False)
    write_dev_output(best_model, dev_loader, "dev1.out", dev_lines, tag_to_int)
    
    # Test Lines
    test_lines = read_data("data/test")
    sentence_word_mapping, original_sentence_length = test_map_sentences(test_lines, word_to_int)
    test_data = torch.utils.data.TensorDataset(sentence_word_mapping, original_sentence_length)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    write_test_output(best_model, test_loader, "test1.out", test_lines, tag_to_int)
    
    # Best Glove Model
    glove_word_embedding = get_glove_word_embedding(words, word_to_int)
    best_glove_model = BiLSTM(embedding_size = 100, hidden_size = 256, total_words = len(words)+2, linear_out_dim=128, embedding=glove_word_embedding).to(device)
    best_glove_model.load_state_dict(torch.load("model/blstm2.pt"))
    
    write_dev_output(best_glove_model, dev_loader, "dev2.out", dev_lines, tag_to_int)
    write_test_output(best_glove_model, test_loader, "test2.out", test_lines, tag_to_int)


# ## 1: Simple Bidirectional LSTM model

def train_simple_blstm():
    # Preparing Training Data
    lines = read_data("data/train")
    words, tags, word_to_int, tag_to_int = get_words_tags_mapping(lines)
    sentence_word_mapping, sentence_tag_mapping, original_sentence_length = map_sentences(lines, word_to_int, tag_to_int)
    train_data = torch.utils.data.TensorDataset(sentence_word_mapping, sentence_tag_mapping, original_sentence_length)
    loader = torch.utils.data.DataLoader(train_data, batch_size=2, shuffle=True)

    # Dev Lines
    dev_lines = read_data("data/dev")
    sentence_word_mapping, sentence_tag_mapping, original_sentence_length = map_sentences(dev_lines, word_to_int, tag_to_int)
    dev_data = torch.utils.data.TensorDataset(sentence_word_mapping, sentence_tag_mapping, original_sentence_length)
    dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=1, shuffle=False)

    # Define the model
    model = BiLSTM(embedding_size = 100, hidden_size = 256, total_words = len(words)+2, linear_out_dim=128, embedding=torch.rand(len(words)+2,100)).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=0.0005)
    lossfunction = torch.nn.CrossEntropyLoss()
    print(model)

    # Training Model
    model = train_bi_lstm(model, optimizer, lossfunction, scheduler, loader, dev_loader, device, n_epochs = 25)
    write_output(model, dev_loader, "dev1.out", dev_lines, tag_to_int)

    # Saving Model
    torch.save(model.state_dict(), 'model/model_weights.pt')
    #torch.save(model.state_dict(), 'model/blstm1.pt')


# ##  2: Using GloVe word embeddings

def train_glove_blstm():
    # Preparing Training Data
    lines = read_data("data/train")
    words, tags, word_to_int, tag_to_int = get_words_tags_mapping(lines)
    sentence_word_mapping, sentence_tag_mapping, original_sentence_length = map_sentences(lines, word_to_int, tag_to_int)
    train_data = torch.utils.data.TensorDataset(sentence_word_mapping, sentence_tag_mapping, original_sentence_length)
    loader = torch.utils.data.DataLoader(train_data, batch_size=2, shuffle=True)
    glove_word_embedding = get_glove_word_embedding(words, word_to_int)

    # Dev Lines
    dev_lines = read_data("data/dev")
    sentence_word_mapping, sentence_tag_mapping, original_sentence_length = map_sentences(dev_lines, word_to_int, tag_to_int)
    dev_data = torch.utils.data.TensorDataset(sentence_word_mapping, sentence_tag_mapping, original_sentence_length)
    dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=1, shuffle=False)

    # Define the model
    glove_model = BiLSTM(embedding_size = 100, hidden_size = 256, total_words = len(words)+2, linear_out_dim=128, embedding=glove_word_embedding).to(device)
    print(glove_model)
    optimizer = torch.optim.SGD(glove_model.parameters(), lr = 0.25)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=0.005)
    lossfunction = torch.nn.CrossEntropyLoss()

    #Training the model
    glove_model = train_bi_lstm(glove_model, optimizer, lossfunction, scheduler, loader, dev_loader, device, n_epochs = 25)
    write_output(glove_model, dev_loader, "dev2.out", dev_lines, tag_to_int)

    # Saving Model
    torch.save(glove_model.state_dict(), 'model/model_weights_glove.pt')
    
    # Saving Model
    #torch.save(glove_model.state_dict(), 'model/blstm2.pt')


if __name__ == "__main__":

    if torch.cuda.is_available():  
        device = "cuda:0" 
    else:  
        device = "cpu"
    
    # Uncomment this to train simple BLSTM
    #train_simple_blstm()

    # Uncomment this to train BLSTM with GloVe
    #train_glove_blstm()

    write_dev_test_on_best()