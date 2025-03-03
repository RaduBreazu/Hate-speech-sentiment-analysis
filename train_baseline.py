from torch.utils.data import Dataset,DataLoader
import pandas as pd
from preprocess_data import load_sentiment_analysis_dataset,load_hate_speech_dataset
import argparse
from transformers import BertTokenizer, BertModel, DistilBertModel
import lightning as pl
import torch.nn as nn
import torch
from aim.pytorch_lightning import AimLogger
from torchmetrics.classification import MulticlassAccuracy,MulticlassF1Score,BinaryF1Score,BinaryAccuracy
import argparse
import transformers


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

acc_sent = BinaryAccuracy(ignore_index=tokenizer.pad_token_id).to(device)
acc_hate = MulticlassAccuracy(num_classes=3,ignore_index=tokenizer.pad_token_id).to(device)
f1_sent = BinaryF1Score(ignore_index=tokenizer.pad_token_id).to(device)
f1_hate = MulticlassF1Score(num_classes=3,ignore_index=tokenizer.pad_token_id).to(device)
map_labels_type_data={
    2:"sent",
    3:"hate"
}

class BERTSeqClassif(nn.Module):
    def __init__(self,num_labels,freeze_weights=False):
        super().__init__()
        self.baseline_bert = BertModel.from_pretrained("bert-base-uncased",torch_dtype=torch.bfloat16)
        if(freeze_weights==True):
            for param in self.baseline_bert.parameters():
                param.requires_grad=False
        self.dropout = nn.Dropout(p=0.2)
        self.classifier_head = nn.Linear(self.baseline_bert.config.hidden_size,num_labels,dtype=torch.bfloat16)
    def forward(self,batch):
        content = batch[0]
        tokenized_content = tokenizer(content,return_tensors="pt",truncation=True,padding='longest',max_length=128)
        out_bert = self.baseline_bert(input_ids=tokenized_content["input_ids"].to(device),attention_mask=tokenized_content["attention_mask"].to(device))
        y_hat = self.classifier_head(self.dropout(out_bert.pooler_output))
        return y_hat

class BaselineModule(pl.LightningModule):
    def __init__(self,num_labels,total_steps,lr,batch_size,warmup_steps,sch_type,freeze_weights=False):
        super().__init__()
        self.save_hyperparameters()
        self.BERT_seq_classif = BERTSeqClassif(num_labels,freeze_weights)
        self.acc_fn = acc_sent if num_labels==2 else acc_hate
        self.f1_fn = f1_sent if num_labels==2 else f1_hate
        self.num_labels=num_labels
        self.total_steps = total_steps
        self.warmup_steps=warmup_steps
        self.sch_type =sch_type
        self.lr=lr
        self.batch_size=batch_size
        self.automatic_optimization = False
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    def compute_metrics(self,y_hat,labels,mode):
        preds_probs = torch.softmax(y_hat,dim=1)
        preds = torch.argmax(preds_probs,dim=1)
        self.log(f"{mode}_accuracy_{map_labels_type_data[self.num_labels]}",self.acc_fn(preds,labels),batch_size=self.batch_size,on_step=False,on_epoch=True)
        self.log(f"{mode}_f1_{map_labels_type_data[self.num_labels]}",self.f1_fn(preds,labels),batch_size=self.batch_size,on_step=False,on_epoch=True)
    def compute_loss(self,batch,mode):
        labels = torch.tensor(batch[1]).to(device)
        y_hat = self.BERT_seq_classif(batch)
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
        self.log("Train_loss",loss_total,batch_size=self.batch_size,on_step=False,on_epoch=True)
        return loss_total
    def validation_step(self,batch,batch_idx):
        loss_total = self.compute_loss(batch,"Test")
        self.log("Test_loss",loss_total,batch_size=self.batch_size,on_step=False,on_epoch=True)
    def configure_optimizers(self):
        #optimizer = torch.optim.SGD(self.parameters(),lr=1e-3,momentum=0.9,weight_decay=0.01)
        optimizer = torch.optim.Adam(self.parameters(),lr=self.lr)
        #scheduler = transformers.get_linear_schedule_with_warmup(optimizer,500,self.dataloader_len*num_epochs)
        scheduler = transformers.get_scheduler(self.sch_type,optimizer,self.warmup_steps,self.total_steps)
        #return optimizer
        return [optimizer],[scheduler]

if __name__=='__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--num_epochs")
    arg_parser.add_argument("--batch_size")
    arg_parser.add_argument("--sch_type",choices=["linear","cosine","inverse_sqrt"],default="linear")
    arg_parser.add_argument("--lr")
    arg_parser.add_argument("--dataset",choices=["hate","sent"])
    arg_parser.add_argument("--warmup_steps")
    parsed_args = arg_parser.parse_args()
    num_epochs = int(parsed_args.num_epochs)
    batch_size = int(parsed_args.batch_size)
    sch_type = parsed_args.sch_type
    lr = float(parsed_args.lr)
    dataset_type = parsed_args.dataset
    warmup_steps = int(parsed_args.warmup_steps)
    seed_value=42
    pl.seed_everything(seed_value)
    if(dataset_type=="hate"):
        train_df,test_df = load_hate_speech_dataset(batch_size=batch_size)
        trainLoader = DataLoader(train_df,batch_size=batch_size,shuffle=True)
        testLoader = DataLoader(test_df,batch_size=batch_size,shuffle=False)
        num_labels = 3
        freeze_weights=False
    else:
        train_df,test_df =load_sentiment_analysis_dataset(batch_size=batch_size)
        trainLoader = DataLoader(train_df,batch_size=batch_size,shuffle=True)
        testLoader = DataLoader(test_df,batch_size=batch_size,shuffle=False)
        num_labels = 2
        freeze_weights = True

    MTLlog = AimLogger(experiment=f"Baseline_BERT_{dataset_type}",train_metric_prefix="Train_",val_metric_prefix="Test_")
    MTLlog.log_hyperparams({
        "sch_type":sch_type,
        "batch_size":batch_size,
        "lr":lr,
        "num_epochs":num_epochs
    })
    csv_logger = pl.pytorch.loggers.csv_logs.CSVLogger(f"Baseline_{dataset_type}")
    checkpoint_callback = pl.pytorch.callbacks.ModelCheckpoint(dirpath=f"Baseline_Model_{dataset_type}_CKPT")
    pl_model = BaselineModule(num_labels,len(trainLoader)* num_epochs,lr,batch_size,warmup_steps,sch_type,freeze_weights=freeze_weights)
    trainer = pl.Trainer(max_epochs=num_epochs,logger=[csv_logger,MTLlog],callbacks=[checkpoint_callback],precision="bf16")
    trainer.fit(pl_model,trainLoader,testLoader)