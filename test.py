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

# test set loading and pre-processing
with open('ted-talks-corpus/test.en','r') as file:
    en_test=file.readlines()
with open('ted-talks-corpus/test.fr','r') as file:
    fr_test=file.readlines()
en_test=add_start_end_toks([tokenise_txt(s,'english') for s in en_test])
fr_test=add_start_end_toks([tokenise_txt(s,'french') for s in fr_test])

en_len_list_test=[len(s) for s in en_test]
fr_len_list_test=[len(s) for s in fr_test]

en_padded_test=[s+(max_len_en-len(s))*['<pad>'] for s in en_test]
fr_padded_test=[s+(max_len_fr-len(s))*['<pad>'] for s in fr_test]
en_wordind_test=[[word_to_ind_en[w] if w in word_to_ind_en else word_to_ind_en['<unk>'] for w in s] for s in en_padded_test]
fr_wordind_test=[[word_to_ind_fr[w] if w in word_to_ind_fr else word_to_ind_fr['<unk>'] for w in s] for s in fr_padded_test]


# defining test dataset and test dataloader
langtrans_dataset_test=LT_Dataset(en_wordind_test,fr_wordind_test,en_len_list_test,fr_len_list_test)
test_dataloader = DataLoader(langtrans_dataset_test, batch_size=batch_size)

# defining the model
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

# loading pre-trained model weights and configuration
checkpoint = torch.load('Transformer.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# running inference on test set, printing avg bleu_score
total_bleu_score=0
total_sen=0
with torch.no_grad():
        loss_tot=0
        for m, (src,trg,x_len,y_len) in enumerate(test_dataloader):
            src=src.to(device)
            trg=trg.to(device)
            output = model(src, trg[:, :-1])
            out_temp=torch.argmax(output,dim=2)
            for l in range(trg.size(0)):
                bleu_score = calculate_bleu(out_temp[l],trg[l,1:y_len[l]-1],vocab_fr,word_to_ind_fr)
                total_bleu_score+=bleu_score
            total_sen+=output.size(0)
print(f"Average Test BLEU Score: {total_bleu_score/total_sen}")