import torch.nn as nn
import torch
from torchvision.models import resnet50, ResNet50_Weights 
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
from dataPipeline.embeddingUtils import embPipeline

class model(nn.Module):
    def __init__(self,backbone, hidden_dim=256, gnn_dim=128, num_classes=8):
        super(model, self).__init__()
        self.backBone = backbone # CNN feature extractor
        inEmbed=0
        outEmbed=0
        self.embed = nn.Sequential(
            nn.Linear(in_features=inEmbed, out_features=outEmbed),
            nn.ReLU(),
        )

        # GNN Block
        self.gnn_block = GraphResidualBlock(in_channels=gnn_dim, out_channels=gnn_dim)

        # Transformer encoder 
        bert_config = BertConfig(
            hidden_size=gnn_dim,
            num_attention_heads=4,
            num_hidden_layers=2,
            intermediate_size=gnn_dim * 4,
            max_position_embeddings=128,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1
        )
        self.transformer = BertModel(bert_config)

        # Classifier on [CLS] token
        self.classifier = nn.Linear(gnn_dim, num_classes)
        return 
    


    def forward(self,data):
        im = data[0]
        G = data[1] 
        # Backbone pass

        extractedFeatures = self.backBone(im)

        # embeddings computations

        embeddings = embPipeline(extractedFeatures,g)
        embeddings = torch.tensor(embeddings)
        embeddings=self.embed(embeddings)

        # GNN pass (for each sample in the batch)
        x_gnn = []
        for i in range(B):
            x = self.gnn_block(node_features[i], edge_index[i])
            x_gnn.append(x)
        x_gnn = torch.stack(x_gnn)  # B x N x gnn_dim

        # token CLS (now all zeros)
        cls_token = torch.zeros(B, 1, x_gnn.size(-1), device=x_gnn.device)
        transformer_input = torch.cat([cls_token, x_gnn], dim=1)  # B x (1 + N) x D

        # transformer pass
        transformer_output = self.transformer(inputs_embeds=transformer_input)
        cls_out = transformer_output.last_hidden_state[:, 0, :]  # B x D

        # classification
        out = self.classifier(cls_out)  # B x num_classes
        return out
    
    def train(self):
        return
    
    def predict(self):
        return
    
    def save(self):
        return

    def load(self):
        return
    

    
class GraphResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, out_channels)
        self.relu = nn.ReLU()
        self.conv2 = GCNConv(out_channels, out_channels)
        self.residual = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        residual = self.residual(x)
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        return x + residual
    

    
def main():
    resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
    modules = list(resnet.children())[:-1]
    featureExtractor = nn.Sequential(*modules)

    mT=model(backbone=featureExtractor)

    return