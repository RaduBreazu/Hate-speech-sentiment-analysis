import random
import argparse
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import lightning as pl
from aim.pytorch_lightning import AimLogger

import transformers
from transformers import BertTokenizer, DistilBertModel

from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

from preprocess_data import load_sentiment_analysis_dataset,load_hate_speech_dataset

from train_baseline import BaselineModule
from preprocess_data import CombinedHSSent

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
acc_sent = MulticlassAccuracy(num_classes = 2).to(device)
acc_hate = MulticlassAccuracy(num_classes = 3).to(device)
f1_sent = MulticlassF1Score(num_classes = 2).to(device)
f1_hate = MulticlassF1Score(num_classes = 3).to(device)


class StudentBERT(nn.Module):
    def __init__(self, num_sentiment_labels : int, num_hate_speech_labels : int):
        super().__init__()
        self.student_model = DistilBertModel.from_pretrained("distilbert-base-uncased", torch_dtype = torch.bfloat16)
        
        self.sentiment_distilled_head = nn.Sequential(
            nn.Dropout(p = 0.2),
            nn.Linear(self.student_model.config.dim, num_sentiment_labels,dtype=torch.bfloat16)
        )
        self.hate_distilled_head = nn.Sequential(
            nn.Dropout(p = 0.2),
            nn.Linear(self.student_model.config.dim, num_hate_speech_labels,dtype=torch.bfloat16)
        )
    
    def forward(self, batch):
        hate_data = batch[0]
        sent_data = batch[1]
        tokenized_hate = tokenizer(hate_data[0],return_tensors="pt",padding='longest',max_length=128,truncation=True)
        tokenized_sent = tokenizer(sent_data[0],return_tensors="pt",padding='longest',max_length=128,truncation=True)
        #print(tokenized_sent["input_ids"].shape)
        sent_out = self.student_model(input_ids = tokenized_sent["input_ids"].to(device), attention_mask = tokenized_sent["attention_mask"].to(device))
        hate_out = self.student_model(input_ids = tokenized_hate["input_ids"].to(device), attention_mask = tokenized_hate["attention_mask"].to(device))
        CLS_sent = sent_out.last_hidden_state[:, 0, :]
        CLS_hate = hate_out.last_hidden_state[:, 0, :]
        y_pred_sent = self.sentiment_distilled_head(CLS_sent)
        y_pred_hate = self.hate_distilled_head(CLS_hate)
        
        return y_pred_sent, y_pred_hate

class TeacherAnnealing(pl.LightningModule):
    def __init__(self, num_sent_labels : int, num_hate_labels : int, total_steps:int,
                 batch_size : int = 16, num_epochs : int = 10):
        super().__init__()
        self.save_hyperparameters()
        self.teacher_sent_model = BaselineModule.load_from_checkpoint("Baseline_Model_sent_CKPT/epoch=9-step=15630-v1.ckpt").BERT_seq_classif
        self.teacher_hate_model = BaselineModule.load_from_checkpoint("Baseline_Model_hate_CKPT/epoch=9-step=12400-v1.ckpt").BERT_seq_classif
        for param in self.teacher_sent_model.parameters():
            param.requires_grad=False
        for param in self.teacher_hate_model.parameters():
            param.requires_grad=False
        self.student_model = StudentBERT(num_sent_labels, num_hate_labels)
        self.loss = nn.CrossEntropyLoss()
        #self.closeness_fn = nn.KLDivLoss(reduction = "batchmean") if additional_loss == "KL" else nn.MSELoss()

        self.batch_size = batch_size
        self.epochs = num_epochs

        # the number of total steps in a single epoch --> 80% of the datasets is used for training (see `preporcess_data.py`)
        self.total_steps = total_steps
        self.hate_labels = num_hate_labels
        self.sent_labels = num_sent_labels

        self.current_step = 0

    def compute_metrics(self, y_pred_sent, sent_labels, y_pred_hate, hate_labels, mode):
        pred_sent = torch.softmax(y_pred_sent,dim=1)
        pred_sent = torch.argmax(pred_sent,dim=1)
        pred_hate = torch.softmax(y_pred_hate,dim=1)
        pred_hate = torch.argmax(pred_hate,dim=1)
        self.log(f"{mode}_accuracy_sent", acc_sent(pred_sent, sent_labels), batch_size = self.batch_size, on_step = False, on_epoch = True)
        self.log(f"{mode}_accuracy_hate", acc_hate(pred_hate, hate_labels), batch_size = self.batch_size, on_step = False, on_epoch = True)
        self.log(f"{mode}_f1_sent", f1_sent(pred_sent, sent_labels), batch_size = self.batch_size, on_step = False, on_epoch = True)
        self.log(f"{mode}_f1_hate", f1_hate(pred_hate, hate_labels), batch_size = self.batch_size, on_step = False, on_epoch = True)

    # Inserted the letter 'λ', because 'lambda' is a keyword in Python (you can just copy-paste the letter if you need it below)
    def compute_total_loss(self, batch, mode : str, λ : float):
        
        sent_labels = torch.tensor(batch[1][1]).to(device)
        hate_labels = torch.tensor(batch[0][1]).to(device)
        y_pred_teacher_sent = self.teacher_sent_model(batch[1])
        y_pred_teacher_hate = self.teacher_hate_model(batch[0])
        y_pred_student_sent, y_pred_student_hate = self.student_model(batch)
        self.compute_metrics(y_pred_student_sent, sent_labels, y_pred_student_hate, hate_labels, mode)
        GT_sent_probabilities = λ * nn.functional.one_hot(sent_labels,num_classes=self.sent_labels) + (1 - λ)*torch.softmax( y_pred_teacher_sent,dim=1)
        GT_sent_probabilities = GT_sent_probabilities.to(dtype=torch.bfloat16)
        #print(torch.softmax( y_pred_teacher_hate,dim=1).shape,y_pred_teacher_hate.shape,nn.functional.one_hot(hate_labels,num_classes=self.sent_labels),nn.functional.one_hot(hate_labels,num_classes=self.sent_labels).shape)
        GT_hate_probabilities = λ * nn.functional.one_hot(hate_labels,num_classes=self.hate_labels) + (1 - λ)*torch.softmax( y_pred_teacher_hate,dim=1)
        GT_hate_probabilities = GT_hate_probabilities.to(dtype=torch.bfloat16)
        #print(y_pred_student_sent.shape)
        #print(y_pred_student_sent)
        #print(GT_sent_probabilities.shape)
        #print(GT_sent_probabilities)
        loss_sent = self.loss(y_pred_student_sent,GT_sent_probabilities)
        loss_hate = self.loss(y_pred_student_hate,GT_hate_probabilities)
        loss_total = loss_sent + loss_hate

        return loss_total
    
    def training_step(self, batch, batch_idx):
        λ = self.current_step / self.total_steps
        result = self.compute_total_loss(batch, "train", λ)
        self.log("Train_loss", result, batch_size = self.batch_size, on_step = False, on_epoch = True)
        self.current_step += 1
        return result
    
    def validation_step(self, batch, batch_idx):
        result = self.compute_total_loss(batch, "test", 1.0) # during testing, we need to compare just with the correct output => we need λ = 1.0
        self.log("Test_loss", result, batch_size = self.batch_size, on_step = False, on_epoch = True)
        return result
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.student_model.parameters(), lr = 1e-4)
    

def main():

    seed_value = 42
    num_epochs =10
    pl.seed_everything(seed_value)
    batch_size=16
    train_sent,test_sent = load_sentiment_analysis_dataset(batch_size=batch_size)
    train_hate,test_hate = load_hate_speech_dataset(batch_size=batch_size)
    trainData = CombinedHSSent(train_hate,train_sent)
    testData = CombinedHSSent(test_hate,test_sent)
    trainLoader = DataLoader(trainData,batch_size=batch_size,shuffle=True)
    testLoader = DataLoader(testData,batch_size=batch_size,shuffle=False)
    
    model = TeacherAnnealing(2, 3, total_steps=len(trainLoader) * num_epochs, batch_size = 16, num_epochs = num_epochs)
    
    logger = AimLogger(experiment = "Teacher_Annealing", train_metric_prefix = "Train_", val_metric_prefix = "Test_")
    csv_logger = pl.pytorch.loggers.csv_logs.CSVLogger("Teacher_annealing")
    # TODO: Add the ModelCheckpoint callback here
    checkpoint_callback = pl.pytorch.callbacks.ModelCheckpoint(dirpath=f"CKPT_Teacher_Annealing")
    
    trainer = pl.Trainer(max_epochs = num_epochs, logger = [csv_logger,logger],callbacks=[checkpoint_callback], precision = "bf16")
    trainer.fit(model, trainLoader, testLoader)

if __name__ == "__main__":
    main()
