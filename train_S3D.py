import itertools
from sklearn.utils import shuffle
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import math, copy, time
from pytorch_model_summary import summary
from Scheduler import CosineAnnealingWarmUpRestarts
from model_test import SLT_Transformer
from dataloader_png import Vocab_tokenizer, get_loader
from bleu import calc_BLEU
from CustomTransform import RandomZeroOut


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(CurrEpoch, Model, iterator, optimizer, scheduler, data_tokenizer,
          metric_translation, metric_gloss, clip, lam_translation = 1, lam_gloss = 5):
    Model.train()
    epoch_loss = 0 ; print("iterator len : ", len(iterator))
    test_sentence = []
    GT_sentence = []

    for i, (frames, glosses, translations) in enumerate(iterator):
        src, orth, trg = \
            frames.to(device), glosses.to(device), translations.to(device)

        optimizer.zero_grad()  # Initialize gradient
        predict_translation, predict_gloss = Model(src, trg[:, :-1])
        translation_dim = predict_translation.shape[-1]
        gloss_dim       = predict_gloss.shape[-1]

        # Generate text file
        for tokens in predict_translation:
            # Get argmax of tokens, bring it back to CPU.
            tokens = torch.argmax(tokens, dim=1).to(dtype=torch.long, device=torch.device("cpu"))
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

        # Predictions
        predict_translation = predict_translation.contiguous().view(-1, translation_dim)
        len_orth_ipt = torch.Tensor([len(orth_) for orth_ in predict_gloss]).type(torch.LongTensor)
        log_prob_orth = predict_gloss.log_softmax(2)

        # GTs
        len_orth = torch.Tensor([(sum(t > 1 for t in gloss)) for gloss in orth]).type(torch.LongTensor)
        orth = orth.type(torch.LongTensor).to(device)
        trg = trg[:, 1:].contiguous().view(-1)
        trg = trg.type(torch.LongTensor).to(device)

        loss_translation = metric_translation(predict_translation, trg)
        loss_gloss = metric_gloss(log_prob_orth.permute(1, 0, 2), orth, len_orth_ipt, len_orth)

        #predict_translation = predict_translation.contiguous().view(-1, translation_dim)
        #predict_gloss = predict_gloss.contiguous().view(-1, gloss_dim)
        # GTs
        # print(orth)
        #orth = orth.contiguous().view(-1)
        #orth = orth.type(torch.LongTensor).to(device)
        #trg = trg[:, 1:].contiguous().view(-1)
        #trg = trg.type(torch.LongTensor).to(device)
        #print(predict_translation.shape, trg.shape)

        #loss_translation = metric(predict_translation, trg)
        #loss_gloss = metric(predict_gloss, orth)

        loss = (lam_translation * loss_translation + lam_gloss * loss_gloss) / (lam_gloss + lam_translation)
        loss.backward()

        # And gradient clipping :
        torch.nn.utils.clip_grad_norm_(Model.parameters(), clip)
        # Update params :
        optimizer.step()
        scheduler.step()
        # total loss in epoch
        epoch_loss += loss.item()

        # Print intra-epoch loss
        if i%1000 == 0:
            print(f'{CurrEpoch} / Step {i} : Loss = {loss.item()}')

    BLEU4 = calc_BLEU(test_sentence, GT_sentence)

    return epoch_loss/len(iterator), BLEU4



def eval(Model, iterator, data_tokenizer, metric_translation, metric_gloss,
         lam_translation = 1, lam_gloss = 5):  # No gradient updatd, no optimizer and clipping
    Model.eval()
    epoch_loss = 0

    print("iterator len : ", len(iterator))
    with torch.no_grad():
        total_len = len(iterator) ; print("iterator len : ", total_len)
        test_sentence = []
        GT_sentence = []

        for i, (frames, glosses, translations) in enumerate(iterator):
            src, orth, trg = \
                frames.to(device), glosses.to(device), translations.to(device)

            predict_translation, predict_gloss = Model(src, trg[:, :-1])

            # Generate text file
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

            # Predictions
            translation_dim = predict_translation.shape[-1]
            predict_translation = predict_translation.contiguous().view(-1, translation_dim)
            len_orth_ipt = torch.Tensor([len(orth_) for orth_ in predict_gloss]).type(torch.LongTensor)
            log_prob_orth = predict_gloss.log_softmax(2)

            # GTs
            len_orth = torch.Tensor([(sum(t > 1 for t in gloss)) for gloss in orth]).type(torch.LongTensor)
            orth = orth.type(torch.LongTensor).to(device)
            trg = trg[:, 1:].contiguous().view(-1)
            trg = trg.type(torch.LongTensor).to(device)

            loss_translation = metric_translation(predict_translation, trg)
            loss_gloss = metric_gloss(log_prob_orth.permute(1, 0, 2), orth, len_orth_ipt, len_orth)

            loss = (lam_translation * loss_translation + lam_gloss * loss_gloss) / (lam_gloss + lam_translation)
            epoch_loss += loss.item()
        print(test_sentence, GT_sentence)
        BLEU4 = calc_BLEU(test_sentence, GT_sentence)

        return epoch_loss / len(iterator), BLEU4



def translate(Model, iterator, metric, data_tokenizer, max_len = 55):
    Model.eval()
    with torch.no_grad():
        test_sentence = []
        GT_sentence = []
        for i, (features, glosses, translations) in enumerate(iterator):
            src, orth, trg = \
                features.to(device), glosses.to(device), translations.to(device)

            S3D_feature = Model.S3D(src)
            src_mask = Model.make_source_mask(S3D_feature)
            enc_feature, predict_gloss = Model.Encoder(S3D_feature, src_mask)

            trg_index = [[data_tokenizer.stoi["<SOS>"]] for i in range(src.size(0))]
            #print(trg_index)
            for j in range(max_len):
                #print(torch.LongTensor(trg_index).shape)
                trg_tensor = torch.LongTensor(trg_index).to(device)
                trg_mask = Model.make_target_mask(trg_tensor)
                output   = Model.Decoder(trg_tensor, enc_feature, src_mask, trg_mask)
                output   = nn.Softmax(dim=-1)(output)

                pred_token = torch.argmax(output, dim=-1)[:,-1]
                #print(torch.argmax(output, dim=-1))

                for target_list, pred in zip(trg_index, pred_token.tolist()):
                    target_list.append(pred)

            # Generate text file
            for tokens in trg_index:
                # Get argmax of tokens, bring it back to CPU.
                #print(tokens)
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

            #print(torch.Tensor(trg_index).shape)

        #print(test_sentence, '\n', GT_sentence)
        BLEU4 = calc_BLEU(test_sentence, GT_sentence)

    return BLEU4


'''
            # Predictions
        len_trans_ipt = torch.Tensor([len(trans) for trans in predict_translation]).type(torch.LongTensor)
        log_prob_trans = predict_translation.log_softmax(2)
        len_orth_ipt = torch.Tensor([len(orth_) for orth_ in predict_gloss]).type(torch.LongTensor)
        log_prob_orth = predict_gloss.log_softmax(2)

        # GTs
        len_orth = torch.Tensor([(sum(t > 0 for t in gloss)) for gloss in orth]).type(torch.LongTensor)
        orth = orth.type(torch.LongTensor).to(device)
        trg = trg[:,1:]
        len_trans = torch.Tensor([(sum(t > 0 for t in trans)) for trans in trg]).type(torch.LongTensor)
        trg = trg.type(torch.LongTensor).to(device)

        loss_translation = metric_translation(log_prob_trans.permute(1, 0, 2), trg, len_trans_ipt, len_trans)
        loss_gloss = metric_gloss(log_prob_orth.permute(1, 0, 2), orth, len_orth_ipt, len_orth)
'''

###################### Hyperparameters, transformation and dataloaders ######################

# Count parameters and initialize weights
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



# Weight initialization
def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


# Training func
def epoch_time(start, end):
    elapsed_time = end - start
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs



############################### Finally, The training Section ###############################
def main():

    base_path = 'C:/Users/Siryu_sci/2021-MLVU/SLT_project/'

    ###################### Get the csv file that needed in training process ######################
    train_data = pd.read_csv(base_path + "PHOENIX-2014-T.train.corpus.csv", delimiter='|')
    val_data = pd.read_csv(base_path + "PHOENIX-2014-T.dev.corpus.csv", delimiter='|')
    test_data = pd.read_csv(base_path + "PHOENIX-2014-T.test.corpus.csv", delimiter='|')

    Traindata = pd.concat([train_data, val_data])  # Train+dev data
    max_len = 55  # Max translation length

    ############################## Define the tokenizer and build. ##############################
    data_tokenizer = Vocab_tokenizer(freq_th=1, max_len=max_len)
    orth_tokenizer = Vocab_tokenizer(freq_th=1, max_len=max_len+1)

    data_tokenizer.build_vocab(Traindata.translation)
    orth_tokenizer.build_vocab(Traindata.orth)

    # target : Translation, glosses : glosses, labels : filename
    targets = data_tokenizer.numericalize(Traindata.translation)
    glosses = orth_tokenizer.numericalize(Traindata.orth)
    labels = Traindata.name.to_numpy()

    print("Translation : ", targets.shape, len(data_tokenizer),
          "\n", "Glosses : ", glosses.shape, len(orth_tokenizer))  # (7615, 300) 2948

    ############################# Split them into Train and dev set #############################
    labels, targets, glosses = shuffle(labels, targets, glosses, random_state=42)
    Idx = 7140
    train_labels, train_glosses, train_translations = labels[:Idx], glosses[:Idx], targets[:Idx]
    val_labels, val_glosses, val_translations = labels[Idx:], glosses[Idx:], targets[Idx:]
    test_labels       = test_data.name.to_numpy()
    test_glosses      = orth_tokenizer.numericalize(test_data.orth)
    test_translations = data_tokenizer.numericalize(test_data.translation)


    lr_init = 4.e-6
    lr_ = [4.e-5, 1.e-4, 2.5e-5]
    n_layer = [2]
    decay_factor = 0.996
    BATCH_SIZE = 3
    N_epoch = 180
    Clip = 1
    # dropout : 0.15 / 0.25 / 0.3 with lr fixed
    # tr : gloss : 1:1 to 1:5

    transforms_train = transforms.Compose([
        RandomZeroOut(p=0.15)]
    )

    train_loader, train_dataset, pad_idx = get_loader(base_path, train_labels, train_glosses,
                                                      train_translations, n_workers=3, BS=BATCH_SIZE, transform=transforms_train)
    val_loader, val_dataset, pad_idx = get_loader(base_path, val_labels, val_glosses,
                                                  val_translations, n_workers=2, BS=BATCH_SIZE, transform=None)
    test_loader, test_dataset, pad_idx = get_loader(base_path, test_labels, test_glosses,
                                                    test_translations, n_workers=2, BS=BATCH_SIZE, transform=None)


    N_tokens = len(data_tokenizer)   # Since 0 is a blank token for CTC
    N_glosses = len(orth_tokenizer)
    encoder_type = 'S3D_small_no_aug' ; print(encoder_type)

    for lr_max, n_layers in itertools.product(lr_, n_layer):
        ######################### Define the model and auxiliary functions #########################
        best_BLEU4_score                 = -float("inf")
        Trainloss, Valloss, BLUE4_scores = [], [], []
        l_tr, l_orth                     = 1, 10


        Transformer = SLT_Transformer(N_glosses, N_tokens, pad_idx, pad_idx, n_layers=n_layers, dropout=0.2, device=device).cuda()
        Transformer.apply(initialize_weights)
        print(f'The model has {count_parameters(Transformer):,} trainable parameters')

       # if os.path.exists('./' + f'lr_{lr_max}_n{n_layers}_d512_{encoder_type}.pt'):
       #     print("Loading state_dict...")
       #     Transformer.load_state_dict(torch.load(f'lr_7e-05_n2_d512_S+R3D.pt'))

        ######################## Optimizer, Scheduler and Loss functions ########################
        optimizer = torch.optim.AdamW(Transformer.parameters(), lr=lr_init, betas = (0.9, 0.999), eps = 1e-08, weight_decay = 1.e-6)
        scheduler = CosineAnnealingWarmUpRestarts(
            optimizer, T_0=7115 // BATCH_SIZE * 48, T_mult=1, eta_max=lr_max, T_up=7115 // BATCH_SIZE * 8, gamma=0.75) # 다음부턴 optimizer를 바꾸자..

        # LOSS functions! #
        criterion_translation = nn.CrossEntropyLoss().cuda() # nn.CTCLoss(blank=0, reduction='sum').cuda()
        criterion_gloss       = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True).cuda() # nn.CTCLoss( blank=0).cuda() #
        # Print the entire model#
        print(summary(Transformer, torch.randn(1, 224, 3, 232, 200).cuda(),
                      torch.randint(0, 54, (1, 54)).cuda(),show_input=True, show_hierarchical=True))
        print('-'*20, f"lr = {lr_max}, n_layer = {n_layers}, translation : gloss = {l_tr} : {l_orth}", '-'*20)


        for epoch in range(N_epoch):
            start = time.time()
            print(device)
            train_loss, BLEU4_train = train(epoch+1, Transformer, train_loader, optimizer,scheduler, data_tokenizer,
                                            criterion_translation, criterion_gloss, Clip, l_tr, l_orth)
            val_loss, BLUE4_score = eval(Transformer, val_loader, data_tokenizer,
                                            criterion_translation, criterion_gloss, l_tr, l_orth)
            test_BLUE4_score = translate(Transformer, test_loader, criterion_translation, data_tokenizer, max_len=max_len)

            Trainloss.append(train_loss) ; Valloss.append(val_loss) ; BLUE4_scores.append(test_BLUE4_score)

            end = time.time()
            epoch_m, epoch_s = epoch_time(start, end)

            if test_BLUE4_score > best_BLEU4_score and (epoch > 5):
                best_BLEU4_score = test_BLUE4_score
                torch.save(Transformer.state_dict(), f'lr_{lr_max}_n{n_layers}_d512_{encoder_type}.pt')


            print('lr = ', optimizer.param_groups[0]['lr'])

            print(f'Epoch {epoch + 1:02} | Time : {epoch_m}m, {epoch_s}s')
            print(f'\t Train Loss : {train_loss:.3f} | Train PPL : {math.exp(train_loss):.3f}')
            print(f'\t Train BLEU4 : {BLEU4_train:.3f}')
            print(f'\t Val Loss : {val_loss:.3f} | Val PPL : {math.exp(val_loss):.3f}')
            print(f'\t Val BLEU4 : {BLUE4_score:.3f}')
            print(f'\t Test BLEU4 : {test_BLUE4_score:.3f}')

            # Save loss figure #
            x_epoch = range(epoch+1)
            fig, ax = plt.subplots(figsize=(12, 8))
            ax2 = ax.twinx()
            ax.plot(x_epoch, Trainloss, 'r-', label = "train loss")
            ax.plot(x_epoch, Valloss, 'b-', label = 'val loss')
            ax2.plot(x_epoch, BLUE4_scores, 'g-', label = 'BLEU4 score')

            ax.set_xlabel("epoch", fontsize = 13)
            ax.set_ylabel("loss", fontsize = 13)
            ax2.set_ylabel("Test BLEU4", fontsize = 13)

            ax.grid()
            ax.legend() ; ax2.legend()


            plt.savefig(f'lr_{lr_max}_n2_d512_{encoder_type}.png', dpi = 250)
            plt.close(fig)

        # Print and store losses #
        print("translation : gloss = ", f'{l_tr} : {l_orth}')
        print(min(Trainloss), min(Valloss), max(BLUE4_scores))

if __name__  == "__main__":
    main()