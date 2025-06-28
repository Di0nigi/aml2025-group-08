import torch.nn as nn
import torch
from torchvision.models import resnet50, ResNet50_Weights 
from transformers import BertTokenizer, BertModel, BertConfig
from dataPipeline.embeddingUtils import embPipeline
from torch_geometric.nn import GCNConv
from transformers.models.bert.modeling_bert import BertLayer

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
        self.config = BertConfig.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        for i in range(4): ## da controllare
            model.encoder.layer[i] = BertGraphEncoder(self.config)

        #for param in self.bert.parameters():
        #    param.requires_grad = False

        
        self.classifier = nn.Linear(gnn_dim, num_classes)
        return 
    


    def forward(self,data):
        im = data[0]
        g = data[1] 
        # Backbone pass

        extractedFeatures = self.backBone(im)

        # Embeddings computations

        embeddings = embPipeline(extractedFeatures,g)
        embeddings = torch.tensor(embeddings)
        embeddings=self.embed(embeddings)

        # bert pass
        embeddings = self.bert(embeddings,...)
        #hidden_states = self.bert.embeddings(input_ids=0, token_type_ids=0)
        #for i, layer_module in enumerate(self.bert.encoder.layer):
        #    hidden_states = layer_module(hidden_states, attention_mask=None)[0]
        #    if i == 3:  # After 4th layer
        #        hidden_states = self.relu(self.inject_layer(hidden_states))
        
        ''' # GNN pass (for each sample in the batch)
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
        cls_out = transformer_output.last_hidden_state[:, 0, :]  # B x D'''

        # classification
        #out = self.classifier(cls_out)  # B x num_classes
        return #out
    
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

class BertGraphEncoder(BertLayer):
    def __init__(self, config):
        super().__init__(config)

        self.gNN = GraphResidualBlock(0,0)

        for name, param in self.named_parameters():
            if not name.startswith("gNN"):
                param.requires_grad = False

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        
        self_attention_outputs = self.attention(hidden_states, attention_mask, **kwargs)
        attention_output = self_attention_outputs[0]
       
        attention_output = self.gNN(attention_output)

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)

        return (layer_output,)
    

    
def main():
    resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
    modules = list(resnet.children())[:-1]
    featureExtractor = nn.Sequential(*modules)

    mT=model(backbone=featureExtractor)

    return