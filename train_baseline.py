from torch.utils.data import Dataset,DataLoader
import pandas as pd
from preprocess_data import load_sentiment_analysis_dataset,load_hate_speech_dataset
import random
from transformers import BertTokenizer, BertModel, DistilBertModel
import lightning as pl
import torch.nn as nn
import torch
from aim.pytorch_lightning import AimLogger
from torchmetrics.classification import MulticlassAccuracy,MulticlassF1Score
import argparse
import transformers
import math
import re
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
acc_sent = MulticlassAccuracy(num_classes=2).to(device)
acc_hate = MulticlassAccuracy(num_classes=3).to(device)
f1_sent = MulticlassF1Score(num_classes=2).to(device)
f1_hate = MulticlassF1Score(num_classes=3).to(device)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


map_labels_type_data={
    2:"sent",
    3:"hate"
}
num_epochs=20
batch_size = 16
class BaselineModule(pl.LightningModule):
    def __init__(self,num_labels,total_steps):
        super().__init__()
        self.save_hyperparameters()
        self.baseline_bert = BertModel.from_pretrained("bert-base-uncased",torch_dtype=torch.bfloat16)
        self.baseline_bert.train()
        """for name,param in self.baseline_bert.named_parameters():
            digit_in_name = re.findall("[0-9]+",name)
            if(len(digit_in_name)==1):
                if(int(digit_in_name[0])<6):
                    param.requires_grad=False
            print(name,param.requires_grad)"""
        self.dropout = nn.Dropout(p=0.15)
        self.classifier_head = nn.Linear(self.baseline_bert.config.hidden_size,num_labels,dtype=torch.bfloat16)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        self.acc_fn = acc_sent if num_labels==2 else acc_hate
        self.f1_fn = f1_sent if num_labels==2 else f1_hate
        self.num_labels=num_labels
        self.total_steps = total_steps
        self.warmup_steps=4000
        self.automatic_optimization = False
    def compute_metrics(self,y_hat,labels,mode):
        self.log(f"{mode}_accuracy_{map_labels_type_data[self.num_labels]}",self.acc_fn(y_hat,labels),batch_size=batch_size,on_step=False,on_epoch=True)
        self.log(f"{mode}_f1_{map_labels_type_data[self.num_labels]}",self.f1_fn(y_hat,labels),batch_size=batch_size,on_step=False,on_epoch=True)
    def compute_loss(self,batch,mode):
        content = batch[0]
        tokenized_content = tokenizer(content,return_tensors="pt",truncation=True,padding='max_length',max_length=128)
        labels = torch.tensor(batch[1]).to(device)
        out_bert = self.baseline_bert(input_ids=tokenized_content["input_ids"].to(device),attention_mask=tokenized_content["attention_mask"].to(device))
        y_hat = self.classifier_head(self.dropout(out_bert.pooler_output))
        self.compute_metrics(y_hat,labels,mode)
        return self.loss_fn(y_hat,labels)
    def training_step(self,batch,batch_idx):
        opt = self.optimizers()
        sch = self.lr_schedulers()
        opt.zero_grad()
        loss_total = self.compute_loss(batch,"Train")
        self.manual_backward(loss_total)
        opt.step()
        #if((batch_idx + 1) % 500 == 0):
        sch.step()
        self.log("Train_loss",loss_total,batch_size=batch_size,on_step=False,on_epoch=True)
        return loss_total
    def validation_step(self,batch,batch_idx):
        loss_total = self.compute_loss(batch,"Test")
        self.log("Test_loss",loss_total,batch_size=batch_size,on_step=False,on_epoch=True)
    def configure_optimizers(self):
        #optimizer = torch.optim.SGD(self.parameters(),lr=1e-3,momentum=0.9,weight_decay=0.01)
        optimizer = torch.optim.AdamW(self.parameters(),lr=5e-5)
        #scheduler = transformers.get_linear_schedule_with_warmup(optimizer,500,self.dataloader_len*num_epochs)
        scheduler = transformers.get_scheduler("linear",optimizer,0,self.total_steps)
        #return optimizer
        return [optimizer],[scheduler]

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",choices=["hate","sent"])
    parsed_arg = parser.parse_args()
    dataset_type=parsed_arg.dataset
    seed_value=42
    pl.seed_everything(seed_value)
    if(dataset_type=="hate"):
        trainLoader,testLoader = load_hate_speech_dataset()
        num_labels = 3
        #batch_example = next(iter(trainLoader))
        #print(torch.tensor(batch_example[1]))
    else:
        trainLoader,testLoader =load_sentiment_analysis_dataset()
        num_labels = 2

    MTLlog = AimLogger(experiment=f"Baseline_BERT_{dataset_type}",train_metric_prefix="Train_",val_metric_prefix="Test_")
    checkpoint_callback = pl.pytorch.callbacks.ModelCheckpoint(dirpath=f"Baseline_Model_{dataset_type}_CKPT")
    pl_model = BaselineModule(num_labels,len(trainLoader)* num_epochs)
    trainer = pl.Trainer(max_epochs=num_epochs,logger=MTLlog,callbacks=[checkpoint_callback],precision="bf16")
    trainer.fit(pl_model,trainLoader,testLoader)