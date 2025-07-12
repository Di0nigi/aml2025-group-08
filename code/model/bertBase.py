import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights
import evaluation.evaluationUtils as ev
from dataPipeline.embeddingUtils import embPipeline, addClsTkCons
from dataPipeline.dataSet import dataPipeline
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

class ImageToBERTClassifier(nn.Module):
    def __init__(self, device, numClasses=9, dropoutRate=0.2):
        super().__init__()
        self.dev = device
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.featureDim = 512
        self.bertDim = 768
        self.project = nn.Linear(self.featureDim, self.bertDim)
        self.config = BertConfig.from_pretrained("bert-base-uncased")
        self.bert = BertModel(self.config)
        for param in self.bert.parameters():
            param.requires_grad = False
        self.clsToken = nn.Parameter(torch.zeros(1, 1, self.bertDim))
        nn.init.trunc_normal_(self.clsToken, std=0.02)
        self.clsToken = nn.Parameter(self.clsToken, requires_grad=False)
        self.dropout = nn.Dropout(dropoutRate)
        self.classifier = nn.Linear(self.bertDim, numClasses)

    def forward(self, x):
        b = x.size(0)
        feats = self.backbone(x)
        feats = feats.flatten(2).transpose(1, 2)
        feats = self.project(feats)
        cls = self.clsToken.expand(b, -1, -1)
        feats = torch.cat([cls, feats], dim=1)
        out = self.bert(inputs_embeds=feats)
        clsOutput = out.last_hidden_state[:, 0]
        logits = self.classifier(self.dropout(clsOutput))
        return logits

def train(model, trainLoader, valLoader, optimizer, criterion, device, epochs=14, patience=5):
    model.to(device)

    trainLossList, trainAccList = [], []
    valLossList, valAccList = [], []
    predList, targetList = [], []

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(epochs):
        print(f"Epoch [{epoch+1}/{epochs}]")
        model.train()
        trainLoss, trainCorrect, trainTotal = 0.0, 0, 0

        for images, labels in trainLoader:
            images, labels = images.to(device), labels.to(device)
            labels = labels.squeeze(1).argmax(dim=1).long()

            logits = model(images)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            trainLoss += loss.item()
            preds = logits.argmax(dim=1)
            trainCorrect += (preds == labels).sum().item()
            trainTotal += labels.size(0)

        trainAcc = trainCorrect / max(trainTotal, 1)
        avgTrainLoss = trainLoss / max(trainTotal, 1)
        trainLossList.append(avgTrainLoss)
        trainAccList.append(trainAcc)

        print(f"Train Loss: {avgTrainLoss:.4f}, Train Accuracy: {trainAcc*100:.2f}%")

        model.eval()
        valLoss, valCorrect, valTotal = 0.0, 0, 0
        predList = []
        targetList = []

        with torch.no_grad():
            for images, labels in valLoader:
                images, labels = images.to(device), labels.to(device)
                labels = labels.squeeze(1).argmax(dim=1).long()

                logits = model(images)
                loss = criterion(logits, labels)

                valLoss += loss.item()

                preds = logits.argmax(dim=1)
                predList.append(logits)
                targetList.append(labels)

                valCorrect += (preds == labels).sum().item()
                valTotal += labels.size(0)

        valAcc = valCorrect / max(valTotal, 1)
        avgValLoss = valLoss / max(valTotal, 1)
        valLossList.append(avgValLoss)
        valAccList.append(valAcc)

        print(f"Val Loss: {avgValLoss:.4f}, Val Accuracy: {valAcc*100:.2f}%\n")

        if avgValLoss < best_val_loss:
            best_val_loss = avgValLoss
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1} due to no improvement.")
                model.load_state_dict(best_model_state)
                break

    return [trainLossList, trainAccList, valLossList, valAccList], [predList, targetList]


def weightDataSet(dataSet):
    train = dataSet[0][0]
    labelList = []
    for b in train:
        _, label = b
        label = int(torch.argmax(label))
        labelList.append(label)
    labelsTensor = torch.tensor(labelList)
    classCounts = torch.bincount(labelsTensor)
    weights = 1.0 / classCounts.float()
    weights = weights / weights.sum()
    return weights

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = dataPipeline("D:\dionigi\Documents\Python scripts\\aml2025Data\dataNorm", split=0.8, batches=16, classes=9)
    model = ImageToBERTClassifier(numClasses=9, device=device)
    model.to(device)
    classWeights = weightDataSet(data)
    lossFn = nn.CrossEntropyLoss(weight=classWeights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    res = train(model, data[0][0], data[1][0], optimizer, lossFn, device, epochs=14)
    ev.displayData(res)


if __name__ == "__main__":
    main()