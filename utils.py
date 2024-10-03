from nltk.tokenize import word_tokenize
from collections import Counter
from torch.utils.data import Dataset
import torch
from torch import nn
from encoder import Encoder
from decoder import Decoder
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import matplotlib.pyplot as plt

# tokeinse sentences of given language
def tokenise_txt(s,lang):
    s= s.lower()
    s = word_tokenize(s, language=lang)
    return s

# replace words that occur very few number of times by <unk> token to get <unk> embedding while training
def replace_low_frequency_words(sentences, threshold=3):
    word_counts = Counter(word for sentence in sentences for word in sentence)
    replaced_sentences = [['<unk>' if word_counts[word] < threshold else word for word in sentence] for sentence in sentences]
    return replaced_sentences

# add start and stop tokens to each sentence
def add_start_end_toks(sentences,start=['<sos>'],end=['<eos>']):
    return [start+s+end for s in sentences]

# dataset for model
class LT_Dataset(Dataset):
    def __init__(self,en_sent,fr_sent,en_lens,fr_lens):
        self.en_list=en_sent
        self.fr_list=fr_sent
        self.en_lens=en_lens
        self.fr_lens=fr_lens
    def __len__(self):
        return len(self.en_list)
    def __getitem__(self,idx):
        return torch.tensor(self.en_list[idx]),torch.tensor(self.fr_list[idx]), self.en_lens[idx], self.fr_lens[idx]

# transformer class: combining encoder and decoder modules
class Transformer(nn.Module):
    def __init__(self, src_pad_idx, trg_pad_idx, trg_sos_idx, enc_voc_size, dec_voc_size, d_model, n_head, max_len,
                 ffn_hidden, n_layers, drop_prob, device):
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.trg_sos_idx = trg_sos_idx
        self.device = device
        self.encoder = Encoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               enc_voc_size=enc_voc_size,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)

        self.decoder = Decoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               dec_voc_size=dec_voc_size,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        output = self.decoder(trg, enc_src, trg_mask, src_mask)
        return output

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).type(torch.ByteTensor).to(self.device)
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask
    
# BLEU score calculation function
def calculate_bleu(pred, truth,vocab_fr,word_to_ind_fr):
    pred = pred.cpu().detach().numpy().flatten()
    truth = truth.cpu().detach().numpy().flatten()
    preds = [int(item) for item in pred]
    truths = [int(item) for item in truth]
    end_sentence_punctuation = [word_to_ind_fr['.'],word_to_ind_fr['?'],word_to_ind_fr['!']]
    stop_ind = min((preds.index(p) for p in end_sentence_punctuation if p in preds), default=len(preds)-1)
    preds = preds[:stop_ind+1]
    if len(preds)==1:
        preds=2*preds
    pred_sentence = [vocab_fr[w] for w in preds]
    target_sentence = [vocab_fr[w] for w in truths]
    smoother = SmoothingFunction().method4
    bleu_score = sentence_bleu([target_sentence], pred_sentence, smoothing_function=smoother)
    return bleu_score

# graph plotting function
def plot_graph(count,arr,col,x_lab,y_lab,tit):
    plt.figure(figsize=(8, 6))
    plt.plot(range(1,count+1), arr, marker='o', linestyle='-', color=col)
    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    plt.title(tit)
    plt.legend()
    plt.grid(True)
    plt.show()

# test sentence and BLEU score file creation code 
def gen_bleu_txt(pred, truth,vocab_fr,word_to_ind_fr):
    pred = pred.cpu().detach().numpy().flatten()
    truth = truth.cpu().detach().numpy().flatten()
    preds = [int(item) for item in pred]
    truths = [int(item) for item in truth]
    end_sentence_punctuation = [word_to_ind_fr['.'],word_to_ind_fr['?'],word_to_ind_fr['!']]
    stop_ind = min((preds.index(p) for p in end_sentence_punctuation if p in preds), default=len(preds)-1)
    preds = preds[:stop_ind+1]
    if len(preds)==1:
        preds=2*preds
    pred_sentence = [vocab_fr[w] for w in preds]
    target_sentence = [vocab_fr[w] for w in truths]
    smoother = SmoothingFunction().method4
    bleu_score = sentence_bleu([target_sentence], pred_sentence, smoothing_function=smoother)
    return bleu_score,pred_sentence