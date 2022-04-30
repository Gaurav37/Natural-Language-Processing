# -*- coding: utf-8 -*-
"""Roberta_Squad_attempt1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/13VgXm69p4oA4SckdOQPQMJuJzr9DiZ7H
"""

# !nvidia-smi

# !pip install pytorch-lightning
# !pip install transformers
# !pip install datasets
# !pip install sentencepiece
# !pip install wandb

# !mkdir squad
# !wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json -O squad/train-v2.0.json
# !wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json -O squad/dev-v2.0.json

import datasets
import torch
import transformers as tfs
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.metrics import Accuracy
from pytorch_lightning import loggers as pl_loggers
import json
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import AdamW
from tqdm.notebook import tqdm
from torch.utils.tensorboard import SummaryWriter

def read_squad(path):
    path = Path(path)
    with open(path, 'rb') as f:
        squad_dict = json.load(f)

    contexts = []
    questions = []
    answers = []
    qac=[]
    counter=0
    for group in squad_dict['data']:
        for passage in group['paragraphs']:
            context = passage['context']  
            for qa in passage['qas']:
                question = qa['question']
                q_answers = qa['answers'].copy()
                q_answers = list(map(lambda x:x['text'], q_answers))
                for answer in qa['answers']:
                    contexts.append(context)
                    questions.append(question)
                    answers.append(answer)
                    qac.append({'context':context,'question':question,'answers':q_answers})
    return contexts, questions, answers, qac

train_contexts, train_questions, train_answers,train_qac = read_squad('squad/train-v2.0.json')
val_contexts, val_questions, val_answers, val_qac = read_squad('squad/dev-v2.0.json')

# train_contexts =  train_contexts[0:100]
# train_questions =  train_questions[0:100]
# train_answers =  train_answers[0:100]
# val_contexts =  val_contexts[0:10]
# val_questions =  val_questions[0:10]
# val_answers =  val_answers[0:10]
# val_qac = val_qac[0:10]

def add_end_idx(answers, contexts):
    for answer, context in zip(answers, contexts):
        gold_text = answer['text']
        start_idx = answer['answer_start']
        end_idx = start_idx + len(gold_text)

        # sometimes squad answers are off by a character or two – fix this
        if context[start_idx:end_idx] == gold_text:
            answer['answer_end'] = end_idx
        elif context[start_idx-1:end_idx-1] == gold_text:
            answer['answer_start'] = start_idx - 1
            answer['answer_end'] = end_idx - 1     # When the gold label is off by one character
        elif context[start_idx-2:end_idx-2] == gold_text:
            answer['answer_start'] = start_idx - 2
            answer['answer_end'] = end_idx - 2     # When the gold label is off by two characters

add_end_idx(train_answers, train_contexts)
add_end_idx(val_answers, val_contexts)

tokenizer = tfs.AutoTokenizer.from_pretrained('roberta-base')


train_encodings = tokenizer(train_contexts, train_questions, truncation=True, padding=True)
val_encodings = tokenizer(val_contexts, val_questions, truncation=True, padding=True)

def add_token_positions(encodings, answers):
    start_positions = []
    end_positions = []
    for i in range(len(answers)):
        start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
        end_positions.append(encodings.char_to_token(i, answers[i]['answer_end'] - 1))
        # if None, the answer passage has been truncated
        if start_positions[-1] is None:
            start_positions[-1] = tokenizer.model_max_length
        if end_positions[-1] is None:
            end_positions[-1] = tokenizer.model_max_length
    encodings.update({'start_positions': start_positions, 'end_positions': end_positions})

add_token_positions(train_encodings, train_answers)
add_token_positions(val_encodings, val_answers)

class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

train_dataset = SquadDataset(train_encodings)
val_dataset = SquadDataset(val_encodings)

def normalize_text(s):
    """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
    import string, re

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_exact_match(prediction, truth):
    return int(normalize_text(prediction) == normalize_text(truth))

def compute_f1(prediction, truth):
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(truth).split()
    
    # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
    
    common_tokens = set(pred_tokens) & set(truth_tokens)
    
    # if there are no common tokens then f1 = 0
    if len(common_tokens) == 0:
        return 0
    
    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)
    
    return 2 * (prec * rec) / (prec + rec)

model = tfs.RobertaForQuestionAnswering.from_pretrained('roberta-base')


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model.to(device)
model.train()

train_loader = DataLoader(train_dataset, batch_size=8)
val_dataloader = torch.utils.data.DataLoader(val_dataset,batch_size=8,shuffle=False)

optim = AdamW(model.parameters(), lr=5e-5)

def calculate_stats(input_ids,start,end,idx):
  batch_start = 8*idx
  batch_end = batch_start+8
  data = val_qac[batch_start:batch_end]
  em = 0
  ef1 = 0
  for i,d in enumerate(data):
    answer_start = start[i]
    answer_end = end[i]
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[i][answer_start:answer_end]))
    gold_ans = d['answers']
    if len(gold_ans)==0:
      gold_ans.append("")
    em_s= max((compute_exact_match(answer, g_answer)) for g_answer in gold_ans)
    ef1_s = max((compute_f1(answer, g_answer)) for g_answer in gold_ans)
    # print(em_s,ef1_s,answer,gold_ans)
    em+=em_s
    ef1+=ef1_s
  return em,ef1



num_epochs = 5

writer = SummaryWriter()

for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)
    model.train()
    running_loss = 0.0
    tk0 = tqdm(train_loader, total=int(len(train_loader)))    
    counter = 0
    for idx,batch in enumerate(tk0):
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        start_positions = batch['start_positions'].to(device)
        end_positions = batch['end_positions'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
        loss = outputs[0]
        loss.backward()
        optim.step()
        running_loss += loss.item() *  batch['input_ids'].size(0)
        counter += 1
        tk0.set_postfix(loss=(running_loss / (counter * train_loader.batch_size)))
    epoch_loss = running_loss / len(train_loader)
    writer.add_scalar('Train/Loss', epoch_loss,epoch)
    print('Training Loss: {:.4f}'.format(epoch_loss))

    model.eval()
    running_val_loss=0
    running_val_em=0
    running_val_f1=0
    tk1 = tqdm(val_dataloader, total=int(len(val_dataloader)))  
    for idx,batch in enumerate(tk1):
      input_ids = batch['input_ids'].to(device)
      attention_mask = batch['attention_mask'].to(device)
      start_positions = batch['start_positions'].to(device)
      end_positions = batch['end_positions'].to(device)
      outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
      running_val_loss += loss.item() *  batch['input_ids'].size(0)
      counter += 1
      tk1.set_postfix(loss=(running_loss / (counter * val_dataloader.batch_size)))
      answer_start = torch.argmax(outputs['start_logits'], dim=1)  
      answer_end = torch.argmax(outputs['end_logits'], dim=1) + 1 
      # print(answer_start)
      # print(answer_end)
      em_score, f1_score = calculate_stats(input_ids,answer_start,answer_end,idx)
      running_val_em += em_score
      running_val_f1 += f1_score
    l = len(val_qac)
    epoch_v_loss = running_val_loss /l
    epoch_v_em = running_val_em/l
    epoch_val_f1 = running_val_f1/l
    writer.add_scalar('Val/Loss', epoch_v_loss,epoch)
    writer.add_scalar('Val/EM', epoch_v_em,epoch)
    writer.add_scalar('Val/F1', epoch_val_f1,epoch)
    print('Val Loss: {:.4f}, EM: {:.4f}, F1: {:.4f} '.format(epoch_v_loss,epoch_v_em,epoch_val_f1))

torch.save(model,'./model_5.pt')

!cp /content/model_5.pt /content/drive/MyDrive/

# Commented out IPython magic to ensure Python compatibility.
# %load_ext tensorboard
# %tensorboard --logdir runs



