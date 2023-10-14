from model import SLT_Transformer
from dataloader import Vocab_tokenizer, get_loader
from sklearn.utils import shuffle
from bleu import calc_BLEU
import pandas as pd
import os

import numpy as np
import torch
import torch.nn as nn


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def eval(Model, iterator, metric, data_tokenizer, trial):  # No gradient updatd, no optimizer and clipping
    Model.eval()
    epoch_loss = 0

    with torch.no_grad():
        total_len = len(iterator)
        test_sentence = []
        GT_sentence   = []

        for i, (features, glosses, translations) in enumerate(iterator):
            src, orth, trg = \
                features.to(device), glosses.to(device), translations.to(device)

            predict_translation, predict_gloss = Model(src, trg[:, :-1])

            for tokens in predict_translation:
                # Get argmax of tokens, bring it back to CPU.
                tokens = torch.argmax(tokens, dim = 1).to(dtype = torch.long, device = torch.device("cpu"))
                tokens = tokens.numpy()
                # make string, append it to test_sentence
                itos = data_tokenizer.stringnize(tokens)
                pred_string = ' '.join(itos)
                test_sentence.append(pred_string)
            for tokens in trg:
                tokens = tokens.to(dtype=torch.long, device=torch.device("cpu"))
                tokens = tokens.numpy()
                # make string, append it to test_sentence
                itos = data_tokenizer.stringnize(tokens[1:])
                GT_string = ' '.join(itos)
                GT_sentence.append(GT_string)


            translation_dim = predict_translation.shape[-1]
            gloss_dim       = predict_gloss.shape[-1]

            # Predictions
            predict_translation = predict_translation.contiguous().view(-1, translation_dim)
            predict_gloss = predict_gloss.contiguous().view(-1, gloss_dim)
            # GTs
            orth = orth.contiguous().view(-1)
            orth = orth.type(torch.LongTensor).to(device)
            trg = trg[:, 1:].contiguous().view(-1)
            trg = trg.type(torch.LongTensor).to(device)

            loss_translation = metric(predict_translation, trg)
            loss_gloss = metric(predict_gloss, orth)
            loss = loss_translation
            epoch_loss += loss.item()

        BLEU4 = calc_BLEU(test_sentence, GT_sentence)

        with open(f"./bestmodel/TestPred_trial_{trial}.txt", "w", -1, "utf-8") as f:
            f.write('\n'.join(test_sentence))
        f.close()
        with open(f"./bestmodel/TestGT_trial_{trial}.txt", "w", -1, "utf-8") as f:
            f.write('\n'.join(GT_sentence))
        f.close()

    return epoch_loss / len(iterator), BLEU4

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def main():
    base_path = 'C:/Users/Siryu_sci/2021-MLVU/SLT_project/'
    train_data = pd.read_csv(base_path + "PHOENIX-2014-T.train.corpus.csv", delimiter='|')
    val_data = pd.read_csv(base_path + "PHOENIX-2014-T.dev.corpus.csv", delimiter='|')
    test_data = pd.read_csv(base_path + "PHOENIX-2014-T.test.corpus.csv", delimiter='|')

    Traindata = pd.concat([train_data, val_data])
    max_len   = 55

    # Define the tokenizer. data : translation, orth : gloss
    data_tokenizer = Vocab_tokenizer(freq_th=1, max_len = max_len)
    orth_tokenizer = Vocab_tokenizer(freq_th=1, max_len = max_len+1)

    data_tokenizer.build_vocab(Traindata.translation)
    orth_tokenizer.build_vocab(Traindata.orth)
    #print(orth_tokenizer.stoi)

    targets = data_tokenizer.numericalize(Traindata.translation)
    glosses = orth_tokenizer.numericalize(Traindata.orth)
    labels  = Traindata.name.to_numpy()

    print("Translation : ", targets.shape, len(data_tokenizer),
          "\n", "Glosses : ", glosses.shape, len(orth_tokenizer))    # (7615, 300) 2948

    ############################# Split them into Train and dev set #############################
    labels, targets, glosses = shuffle(labels, targets, glosses, random_state=42)

    train_labels, train_glosses, train_translations = labels[:7115], glosses[:7115], targets[:7115]
    val_labels, val_glosses, val_translations = labels[7115:], glosses[7115:], targets[7115:]
    test_labels = test_data.name.to_numpy()
    test_glosses = orth_tokenizer.numericalize(test_data.orth)
    test_translations = data_tokenizer.numericalize(test_data.translation)

    BATCH_SIZE = 8

    train_loader, train_dataset, pad_idx = get_loader(base_path, train_labels, train_glosses,
                                                    train_translations, n_workers=2, BS=BATCH_SIZE, transform=None)
    val_loader, val_dataset, pad_idx = get_loader(base_path, val_labels, val_glosses,
                                                val_translations, n_workers=2, BS=BATCH_SIZE, transform=None)
    test_loader, test_dataset, pad_idx = get_loader(base_path, test_labels, test_glosses,
                                                  test_translations, n_workers=2, BS=BATCH_SIZE, transform=None)

    N_tokens = len(data_tokenizer)  # Since we're only training the model on the training dataset!
    N_glosses = len(orth_tokenizer)

    ######################### Define the model and auxiliary functions #########################
    Transformer = SLT_Transformer(N_glosses, N_tokens, pad_idx, pad_idx, device=device).cuda()
    criterion = nn.CrossEntropyLoss().cuda()

    print(f'The model has {count_parameters(Transformer):,} trainable parameters')

    Transformer.load_state_dict(torch.load('lr_7e-05_n2_d512_R3D.pt'))


    total_loss = 0
    N_trial    = 5
    for i in range(N_trial):
        test_loss, BLEU4_score  = eval(Transformer, test_loader, criterion, data_tokenizer, i)
        print('BLEU4 = ', BLEU4_score) ; total_loss+=test_loss

    print("average loss : ", total_loss/N_trial)

if __name__  == "__main__":
    main()