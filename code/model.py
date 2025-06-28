import torch.nn as nn
import torch
from torchvision.models import resnet50, ResNet50_Weights 
from transformers import BertTokenizer, BertModel, BertForSequenceClassification


class model(nn.Module):
    def __init__(self,backbone):
        super(model, self).__init__()
        self.backBone = backbone
        self.encoder = nn.Sequential(
        )
        self.classifier = nn.Sequential(
        )
        return 
    def forward(self,data):
        im = data[0]
        G = data[1] 
        # Backbone pass

        extractedFeatures = self.backBone(im)

        # embeddings computations

        # encoder pass
        
        # classification

        return
    
    def train(self):
        return
    
    def predict(self):
        return
    
    def save(self):
        return

    def load(self):
        return
    

def main():
    resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
    modules = list(resnet.children())[:-1]
    featureExtractor = nn.Sequential(*modules)

    mT=model(backbone=featureExtractor)

    return