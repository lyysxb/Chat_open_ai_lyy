import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, classification_report
from tensorboardX import SummaryWriter

class CustomDataset(Dataset):
    def __init__(self, file_path, tokenizer):
        self.data = []
        self.labels = []
        self.tokenizer = tokenizer

        with open(file_path, 'r', encoding='，utf-8') as file:
            lines = file.readlines()
            for line in lines:
                data_line = line.strip().rsplit('，',maxsplit = 1)
                text = data_line[0]
                label = data_line[-1]
                self.data.append(text)
                self.labels.append(int(label)-1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data[index]
        label = self.labels[index]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def train(model, dataloader, optimizer,  device):
    model.train()
    total_loss = 0

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        # scheduler.step()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def evaluate(model, dataloader, device):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predicted_labels = torch.argmax(logits, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted_labels.cpu().numpy())

    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    return accuracy, report


def main():
    # 设置设备
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    # 加载BERT模型和分词器
    model_name = 'bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=6)
    model.to(device)

    # 加载数据集
    train_dataset = CustomDataset('train.txt', tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=20, shuffle=True)

    test_dataset = CustomDataset('test.txt', tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=20, shuffle=False)

    # 设置优化器和学习率调度器
    optimizer = AdamW(model.parameters(), lr=2e-5)
    # total_steps = len(train_dataloader) * 10
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # 设置Tensorboard日志
    writer = SummaryWriter()

    # 开始训练
    for epoch in range(10):
        train_loss = train(model, train_dataloader, optimizer, device)
        test_accuracy, test_report = evaluate(model, test_dataloader, device)

        # 在Tensorboard中记录训练和测试指标
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/test', test_accuracy, epoch)
        writer.add_text('Classification Report/test', test_report, epoch)

        print(f'Epoch {epoch + 1}: Train Loss = {train_loss}, Test Accuracy = {test_accuracy}')
        torch.save(model.state_dict(), f'bert_model_weights_epoch{epoch + 1}.pth')
    writer.close()


if __name__ == '__main__':
    main()
