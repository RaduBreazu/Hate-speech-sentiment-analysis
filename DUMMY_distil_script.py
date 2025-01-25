#!/usr/bin/env python
# coding: utf-8

# In[49]:


from torch.utils.data import Dataset,DataLoader
import pandas as pd
from preprocess_data import load_sentiment_analysis_dataset,load_hate_speech_dataset
import random
from transformers import BertTokenizer, DistilBertModel
import lightning as pl
import torch.nn as nn
import torch
from aim.pytorch_lightning import AimLogger
from torchmetrics.classification import MulticlassAccuracy,MulticlassF1Score
import transformers
from train_model import MTLModule
import argparse
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
acc_sent = MulticlassAccuracy(num_classes=10).to(device)
acc_hate = MulticlassAccuracy(num_classes=2).to(device)
f1_sent = MulticlassF1Score(num_classes=10).to(device)
f1_hate = MulticlassF1Score(num_classes=2).to(device)


# In[3]:


class HateSpeechSentAnalysis(Dataset):
    def __init__(self,df_hate,df_sent):
        self.content_hate = df_hate["Content"]
        self.content_sent = df_sent["Content"]
        self.label_hate = df_hate["Label"]
        self.label_sent = df_sent["Label"]
        self.size = len(self.content_hate)
    def __getitem__(self,idx):
        return {
            "hate_content":self.content_hate.iloc[idx],
            "hate_label":self.label_hate.iloc[idx],
            "sent_content":self.content_sent.iloc[idx],
            "sent_label":self.label_sent.iloc[idx]
        }
    def __len__(self):
        return self.size


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
def collator(dict_list):
    result_dict={
        "tokenized_hate_content":None,
        "hate_labels":[],
        "tokenized_sent_content":None,
        "sent_labels":[]
    }
    hate_list,sent_list=[],[]
    for D in dict_list:
        hate_list.append(D["hate_content"])
        result_dict["hate_labels"].append(D["hate_label"])
        result_dict["sent_labels"].append(D["sent_label"])
        sent_list.append(D["sent_content"])
    result_dict["tokenized_hate_content"]=tokenizer(hate_list,padding='longest',truncation=True,return_tensors='pt')
    result_dict["tokenized_sent_content"]=tokenizer(sent_list,padding='longest',truncation=True,return_tensors='pt')
    return result_dict


class StudentBERT(nn.Module):
    def __init__(self,num_sentiment_labels,num_hate_speech_labels):
        super().__init__()
        self.student_model = DistilBertModel.from_pretrained("distilbert-base-uncased",torch_dtype=torch.bfloat16)
        
        self.sentiment_distilled_head = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(self.student_model.config.dim, num_sentiment_labels,dtype=torch.bfloat16)
        )
        self.hate_distilled_head = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(self.student_model.config.dim, num_hate_speech_labels,dtype=torch.bfloat16)
        )
    def forward(self,batch_dict):
        tokenized_hate = batch_dict["tokenized_hate_content"]
        tokenized_sent = batch_dict["tokenized_sent_content"]
        sent_out = self.student_model(input_ids=tokenized_sent["input_ids"],attention_mask=tokenized_sent["attention_mask"])
        hate_out = self.student_model(input_ids=tokenized_hate["input_ids"],attention_mask=tokenized_hate["attention_mask"])
        CLS_sent = sent_out.last_hidden_state[:,0,:]
        CLS_hate = hate_out.last_hidden_state[:,0,:]
        y_hat_sent = self.sentiment_distilled_head(CLS_sent)
        y_hat_hate = self.hate_distilled_head(CLS_hate)
        return y_hat_sent,y_hat_hate
        


# In[56]:

batch_size = 16
num_epochs=10
class Traditional_Distillation(pl.LightningModule):
    def __init__(self,num_sentiment_labels,num_hate_speech_labels,additional_loss):
        super().__init__()
        self.save_hyperparameters()
        self.teacher_bert = MTLModule.load_from_checkpoint("Teacher_Model_CKPT/epoch=9-step=15630.ckpt").teacher_bert
        for param in self.teacher_bert.parameters():
            param.requires_grad=False
        self.teacher_bert.eval()
        self.student_bert = StudentBERT(num_sentiment_labels,num_hate_speech_labels)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        self.additional_loss = additional_loss
        self.closeness_fn = nn.KLDivLoss(reduction="batchmean") if additional_loss=="KL" else nn.MSELoss()
    def compute_metrics(self,y_hat_sent,sent_labels,y_hat_hate,hate_labels,mode):
        self.log(f"{mode}_accuracy_sent",acc_sent(y_hat_sent,sent_labels),batch_size=batch_size,on_step=False,on_epoch=True)
        self.log(f"{mode}_accuracy_hate",acc_hate(y_hat_hate,hate_labels),batch_size=batch_size,on_step=False,on_epoch=True)
        self.log(f"{mode}_f1_sent",f1_sent(y_hat_sent,sent_labels),batch_size=batch_size,on_step=False,on_epoch=True)
        self.log(f"{mode}_f1_hate",f1_hate(y_hat_hate,hate_labels),batch_size=batch_size,on_step=False,on_epoch=True)
    def compute_total_loss(self,batch_dict,mode):
        sent_labels = torch.tensor(batch_dict["sent_labels"]).to(device)
        hate_labels = torch.tensor(batch_dict["hate_labels"]).to(device)
        y_hat_teacher_sent,y_hat_teacher_hate = self.teacher_bert(batch_dict)
        y_hat_student_sent,y_hat_student_hate = self.student_bert(batch_dict)
        self.compute_metrics(y_hat_student_sent,sent_labels,y_hat_student_hate,hate_labels,mode)
        loss_sent = self.loss_fn(y_hat_student_sent,sent_labels)
        loss_hate = self.loss_fn(y_hat_student_hate,hate_labels)
        loss_total = loss_sent+loss_hate
        if(self.additional_loss == "KL"):
            y_hat_sent = nn.functional.log_softmax(y_hat_student_sent,dim=1)
            y_hat_hate = nn.functional.log_softmax(y_hat_student_hate,dim=1)
            y_hat_teacher_sent = nn.functional.softmax(y_hat_teacher_sent,dim=1)
            y_hat_teacher_hate = nn.functional.softmax(y_hat_teacher_hate,dim=1)
        else:
            y_hat_sent = y_hat_student_sent
            y_hat_hate = y_hat_student_hate
        loss_total += self.closeness_fn(y_hat_sent,y_hat_teacher_sent)
        loss_total += self.closeness_fn(y_hat_hate,y_hat_teacher_hate)
        
        return loss_total
    def training_step(self,batch_dict,batch_idx):
        #opt = self.optimizers()
        #sch = self.lr_schedulers()
        #opt.zero_grad()
        loss_total = self.compute_total_loss(batch_dict,"Train")
        #self.manual_backward(loss_total)
        #opt.step()
        self.log("Train_loss",loss_total,batch_size=batch_size,on_step=False,on_epoch=True)
        return loss_total
    def validation_step(self,batch_dict,batch_idx):
        loss_total = self.compute_total_loss(batch_dict,"Test")
        self.log("Test_loss",loss_total,batch_size=batch_size,on_step=False,on_epoch=True)
    def configure_optimizers(self):
        #optimizer = torch.optim.SGD(self.parameters(),lr=1e-2,momentum=0.9,weight_decay=0.01)
        optimizer = torch.optim.AdamW(self.parameters(),lr=1e-4,weight_decay=1e-2)
        #scheduler = transformers.get_linear_schedule_with_warmup(optimizer,500,self.dataloader_len*num_epochs)
        return optimizer
        #return [optimizer],[scheduler]

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--additional_loss_type")
    args = parser.parse_args()
    additional_loss_type = args.additional_loss_type
    train_sent,test_sent = pd.read_csv("sent_train.csv"),pd.read_csv("sent_test.csv")
    train_hate,test_hate = pd.read_csv("HS_train.csv"),pd.read_csv("HS_test.csv")
    train_sent.fillna(value="",inplace=True)
    test_sent.fillna(value="",inplace=True)
    trainParallelDataset = HateSpeechSentAnalysis(train_hate,train_sent)
    testParallelDataset = HateSpeechSentAnalysis(test_hate,test_sent)

    seed_value=42
    pl.seed_everything(seed_value)
    
    trainLoader = DataLoader(trainParallelDataset,batch_size=batch_size,shuffle=True,collate_fn=collator)
    testLoader = DataLoader(testParallelDataset,batch_size=batch_size,shuffle=False,collate_fn=collator)

    MTLlog = AimLogger(experiment=f"Teacher_model_train_{additional_loss_type}",train_metric_prefix="Train_",val_metric_prefix="Test_")
    checkpoint_callback = pl.pytorch.callbacks.ModelCheckpoint(dirpath=f"Distillation_Model_CKPT_{additional_loss_type}")
    pl_model = Traditional_Distillation(10,2,additional_loss_type)
    trainer = pl.Trainer(max_epochs=num_epochs,logger=MTLlog,callbacks=[checkpoint_callback])
    trainer.fit(pl_model,trainLoader,testLoader)