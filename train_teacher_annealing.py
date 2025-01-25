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
from train_model import MTLModule
from train_baseline import BaselineModule

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
acc_sent = MulticlassAccuracy(num_classes = 10).to(device)
acc_hate = MulticlassAccuracy(num_classes = 2).to(device)
f1_sent = MulticlassF1Score(num_classes = 10).to(device)
f1_hate = MulticlassF1Score(num_classes = 2).to(device)

class HateSpeechSentAnalysis(Dataset):
    def __init__(self, df_hate, df_sent):
        self.content_hate = df_hate["tweet"]
        self.content_sent = df_sent["Content"]
        self.label_hate = df_hate["class"]
        self.label_sent = df_sent["Label"]
        
        self.size = len(self.content_hate)
    
    def __getitem__(self, idx):
        return {
            "hate_content": self.content_hate.iloc[idx],
            "hate_label": self.label_hate.iloc[idx],
            "sent_content": self.content_sent.iloc[idx],
            "sent_label": self.label_sent.iloc[idx]
        }
    
    def __len__(self):
        return self.size

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
    
    def forward(self, batch_dict):
        tokenized_hate = batch_dict["tokenized_hate_content"]
        tokenized_sent = batch_dict["tokenized_sent_content"]
        sent_out = self.student_model(input_ids = tokenized_sent["input_ids"], attention_mask = tokenized_sent["attention_mask"])
        hate_out = self.student_model(input_ids = tokenized_hate["input_ids"], attention_mask = tokenized_hate["attention_mask"])
        CLS_sent = sent_out.last_hidden_state[:, 0, :]
        CLS_hate = hate_out.last_hidden_state[:, 0, :]
        y_pred_sent = self.sentiment_distilled_head(CLS_sent)
        y_pred_hate = self.hate_distilled_head(CLS_hate)
        
        return y_pred_sent, y_pred_hate

class TeacherAnnealing(pl.LightningModule):
    def __init__(self, dataset_size : int, num_sent_labels : int, num_hate_labels : int,
                 batch_size : int = 16, num_epochs : int = 10, additional_loss : str = "MSE"):
        super().__init__()
        self.save_hyperparameters()
        self.teacher_sent_model = BaselineModule.load_from_checkpoint("CKPT_SENT")
        self.teacher_hate_model = BaselineModule.load_from_checkpoint("CKPT_HATE")
        self.student_model = StudentBERT(num_sent_labels, num_hate_labels)
        self.loss = nn.CrossEntropyLoss(ignore_index = tokenizer.pad_token_id)
        self.closeness_fn = nn.KLDivLoss(reduction = "batchmean") if additional_loss == "KL" else nn.MSELoss()

        self.batch_size = batch_size
        self.epochs = num_epochs

        # the number of total steps in a single epoch --> 80% of the datasets is used for training (see `preporcess_data.py`)
        self.total_steps = int(0.8 * dataset_size) // batch_size

    def compute_metrics(self, y_pred_sent, sent_labels, y_pred_hate, hate_labels, mode):
        self.log(f"{mode}_accuracy_sent", acc_sent(y_pred_sent, sent_labels), batch_size = self.batch_size, on_step = False, on_epoch = True)
        self.log(f"{mode}_accuracy_hate", acc_hate(y_pred_hate, hate_labels), batch_size = self.batch_size, on_step = False, on_epoch = True)
        self.log(f"{mode}_f1_sent", f1_sent(y_pred_sent, sent_labels), batch_size = self.batch_size, on_step = False, on_epoch = True)
        self.log(f"{mode}_f1_hate", f1_hate(y_pred_hate, hate_labels), batch_size = self.batch_size, on_step = False, on_epoch = True)

    # Inserted the letter 'λ', because 'lambda' is a keyword in Python (you can just copy-paste the letter if you need it below)
    def compute_total_loss(self, batch_dict, mode : str, λ : float):
        sent_labels = torch.tensor(batch_dict["sent_labels"]).to(device)
        hate_labels = torch.tensor(batch_dict["hate_labels"]).to(device)
        y_pred_teacher_sent = self.teacher_sent_model(batch_dict)
        y_pred_teacher_hate = self.teacher_hate_model(batch_dict)
        y_pred_student_sent, y_pred_student_hate = self.student_model(batch_dict)
        self.compute_metrics(y_pred_student_sent, sent_labels, y_pred_student_hate, hate_labels, mode)
        loss_sent = self.loss(λ * sent_labels + (1 - λ) * y_pred_teacher_sent, y_pred_student_sent)
        loss_hate = self.loss(λ * hate_labels + (1 - λ) * y_pred_teacher_hate, y_pred_student_hate)
        loss_total = loss_sent + loss_hate

        if self.additional_loss == "KL":
            y_pred_student_sent = F.log_softmax(y_pred_student_sent, dim = 1)
            y_pred_student_hate = F.log_softmax(y_pred_student_hate, dim = 1)
            y_pred_teacher_sent = F.softmax(y_pred_teacher_sent, dim = 1)
            y_pred_teacher_hate = F.softmax(y_pred_teacher_hate, dim = 1)
        
        loss_sent_kl = self.closeness_fn(λ * sent_labels + (1 - λ) * y_pred_teacher_sent, y_pred_student_sent)
        loss_hate_kl = self.closeness_fn(λ * hate_labels + (1 - λ) * y_pred_teacher_hate, y_pred_student_hate)
        loss_total += loss_sent_kl + loss_hate_kl

        return loss_total
    
    def training_step(self, batch, batch_idx):
        λ = batch_idx / self.total_steps
        result = self.compute_total_loss(batch, "train", λ)
        self.log("Train_loss", result, batch_size = self.batch_size, on_step = False, on_epoch = True)
        return result
    
    def validation_step(self, batch, batch_idx):
        result = self.compute_total_loss(batch, "test", 1.0) # during testing, we need to compare just with the correct output => we need λ = 1.0
        self.log("Test_loss", result, batch_size = self.batch_size, on_step = False, on_epoch = True)
        return result
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.student_model.parameters(), lr = 1e-4, weight_decay = 1e-2)
    

def main():
    # Get the additional loss type
    parser = argparse.ArgumentParser()
    parser.add_argument("--additional_loss_type") # for default, use "MSE" for the additional loss
    args = parser.parse_args()
    additional_loss_type = args.additional_loss_type
    
    # Load the datasets
    train_sent, test_sent = pd.read_csv("sent_train.csv"), pd.read_csv("sent_test.csv")
    train_hate, test_hate = pd.read_csv("HS_train.csv"), pd.read_csv("HS_test.csv")
    train_sent.fillna(value = "", inplace = True)
    test_sent.fillna(value = "", inplace = True)
    trainParallelDataset = HateSpeechSentAnalysis(train_hate, train_sent)
    testParallelDataset = HateSpeechSentAnalysis(test_hate, test_sent)

    seed_value = 42
    pl.seed_everything(seed_value)

    train_dataloader = DataLoader(trainParallelDataset, batch_size = 16, shuffle = True)
    test_dataloader = DataLoader(testParallelDataset, batch_size = 16, shuffle = False)
    
    model = TeacherAnnealing(len(train_dataloader), 10, 2, batch_size = 16, num_epochs = 10, additional_loss = additional_loss_type)
    
    logger = AimLogger(experiment = "Teacher_Annealing", train_metric_prefix = "Train_", val_metric_prefix = "Test_")
    # TODO: Add the ModelCheckpoint callback here
    
    trainer = pl.Trainer(max_epochs = 10, logger = logger)
    trainer.fit(model, train_dataloader, test_dataloader)

if __name__ == "__main__":
    main()
