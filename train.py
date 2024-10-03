import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from utils import *

# setting decide as GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# defining hyperparameters
batch_size=24
n_heads=2
dropout=0.2
l_rate=1e-5
model_dim=512
n_layers=4
max_len=800
ffn_hidden=1024

# loading train set
with open('ted-talks-corpus/train.en','r') as file:
    en_train=file.readlines()
with open('ted-talks-corpus/train.fr','r') as file:
    fr_train=file.readlines()

# train set loading and preprocessing
en_train=add_start_end_toks(replace_low_frequency_words([tokenise_txt(s,'english') for s in en_train]))
fr_train=add_start_end_toks(replace_low_frequency_words([tokenise_txt(s,'french') for s in fr_train]))

# vocabs
word_counts_en=Counter([word for sentence in en_train for word in sentence])
vocab_en=[word for word, _ in word_counts_en.items()]+['<pad>']
word_counts_fr=Counter([word for sentence in fr_train for word in sentence])
vocab_fr=[word for word, _ in word_counts_fr.items()]+['<pad>']

# sentence lengths, padding
max_len_en=max([len(s) for s in en_train])
max_len_fr=max([len(s) for s in fr_train])
en_len_list=[len(s) for s in en_train]
fr_len_list=[len(s) for s in fr_train]
word_to_ind_en={}
for i,w in enumerate(vocab_en):
    word_to_ind_en[w]=i
word_to_ind_fr={}
for i,w in enumerate(vocab_fr):
    word_to_ind_fr[w]=i

# padding
en_padded_train=[s+(max_len_en-len(s))*['<pad>'] for s in en_train]
fr_padded_train=[s+(max_len_fr-len(s))*['<pad>'] for s in fr_train]
en_wordind_train=[[word_to_ind_en[w] for w in s] for s in en_padded_train]
fr_wordind_train=[[word_to_ind_fr[w] for w in s] for s in fr_padded_train]

# val set loading and pre-processing
with open('ted-talks-corpus/dev.en','r') as file:
    en_val=file.readlines()
with open('ted-talks-corpus/dev.fr','r') as file:
    fr_val=file.readlines()
en_val=add_start_end_toks([tokenise_txt(s,'english') for s in en_val])
fr_val=add_start_end_toks([tokenise_txt(s,'french') for s in fr_val])

en_len_list_val=[len(s) for s in en_val]
fr_len_list_val=[len(s) for s in fr_val]

en_padded_val=[s+(max_len_en-len(s))*['<pad>'] for s in en_val]
fr_padded_val=[s+(max_len_fr-len(s))*['<pad>'] for s in fr_val]
en_wordind_val=[[word_to_ind_en[w] if w in word_to_ind_en else word_to_ind_en['<unk>'] for w in s] for s in en_padded_val]
fr_wordind_val=[[word_to_ind_fr[w] if w in word_to_ind_fr else word_to_ind_fr['<unk>'] for w in s] for s in fr_padded_val]

# Datasets
langtrans_dataset_train=LT_Dataset(en_wordind_train,fr_wordind_train,en_len_list,fr_len_list)
langtrans_dataset_val=LT_Dataset(en_wordind_val,fr_wordind_val,en_len_list_val,fr_len_list_val)
# Dataloaders
train_dataloader = DataLoader(langtrans_dataset_train, batch_size=batch_size,shuffle=True)
val_dataloader = DataLoader(langtrans_dataset_val, batch_size=batch_size,shuffle=True)

# Init models
torch.autograd.set_detect_anomaly(True)
model = Transformer(src_pad_idx=word_to_ind_en["<pad>"],
                    trg_pad_idx=word_to_ind_fr["<pad>"],
                    trg_sos_idx=word_to_ind_fr["<sos>"],
                    d_model=model_dim,
                    enc_voc_size=len(vocab_en),
                    dec_voc_size=len(vocab_fr),
                    max_len=max_len,
                    ffn_hidden=ffn_hidden,
                    n_head=n_heads,
                    n_layers=n_layers,
                    drop_prob=dropout,
                    device=device)
model=model.to(device)
criterion = nn.CrossEntropyLoss(ignore_index=word_to_ind_fr["<pad>"])
optimizer = optim.Adam(model.parameters(), lr=l_rate)

# train loop
model.train()
epochs=20
train_losses_lst=[]
val_losses_lst=[]
val_bleu_scores=[]
for epoch in range(epochs):
    epoch_loss = 0
    for i, (src,trg,x_len,y_len) in enumerate(train_dataloader):
        src=src.to(device)
        trg=trg.to(device)
        optimizer.zero_grad()
        output = model(src, trg[:, :-1])
        output_reshape = output.contiguous().view(-1, output.shape[-1])
        trg = trg[:, 1:].contiguous().view(-1)
        loss = criterion(output_reshape, trg)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        if (i+1)%100==0:
            print('step :', round((i / len(train_dataloader)) * 100, 2), '% , loss :', loss.item())
    print(f"Epoch: {epoch+1}, Train Loss: {epoch_loss/len(train_dataloader)}")
    train_losses_lst.append(epoch_loss/len(train_dataloader))
    with torch.no_grad():
        loss_tot=0
        total_sen_val=0
        total_bleu_score_val=0
        for m, (src,trg,x_len,y_len) in enumerate(val_dataloader):
            src=src.to(device)
            trg=trg.to(device)
            output = model(src, trg[:, :-1])
            out_temp=torch.argmax(output,dim=2)
            for l in range(trg.size(0)):
                bleu_score = calculate_bleu(out_temp[l],trg[l,1:y_len[l]-1],vocab_fr,word_to_ind_fr)
                total_bleu_score_val+=bleu_score
            total_sen_val+=output.size(0)
            output_reshape = output.contiguous().view(-1, output.shape[-1])
            trg = trg[:, 1:].contiguous().view(-1)
            loss = criterion(output_reshape, trg)
            loss_tot += loss.item()
            if (m+1)%10==0:
                print('step :', round((m / len(val_dataloader)) * 100, 2), '% , loss :', loss.item())
        print(f"Epoch: {epoch+1}, Val Loss: {loss_tot/len(val_dataloader)}, Average Val Set BLEU Score: {total_bleu_score_val/total_sen_val}")
        val_bleu_scores.append(total_bleu_score_val/total_sen_val)
        val_losses_lst.append(loss_tot/len(val_dataloader))

save_path = 'Transformer.pt'
torch.save({
    'epoch': epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict()
}, save_path)