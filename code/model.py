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
        
        self.embed = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=gnn_dim),
            nn.ReLU(),
        )

        # Transformer encoder, inject BERT layers
        self.config = BertConfig.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        self.bert.encoder.layer = nn.ModuleList([
            BertGraphEncoder(self.config) for _ in range(self.config.num_hidden_layers)
        ])

        #for param in self.bert.parameters():
        #    param.requires_grad = False

        self.classifier = nn.Linear(gnn_dim, num_classes)
        
        return 
    


    def forward(self,data):
        im = data[0]
        graph = data[1] 

        # Backbone pass, extract CNN features
        extractedFeatures = self.backBone(im)

        # Embeddings computations
        embeddings = embPipeline(extractedFeatures,graph)
        embeddings = torch.tensor(embeddings)
        embeddings=self.embed(embeddings)

        # bert with injected GNN
        output = self.bert(inputs_embeds=embeddings, attention_mask=None, edge_index=graph.edge_index)

        # CLS loken for classification
        cls_output = output.last_hidden_state[:, 0, :] 
        logits = self.classifier(cls_output)

        return logits
    
    def train(self,dataLoaders,epochs=10):

        trainIm=dataLoaders[0][0]
        trainVe=dataLoaders[0][1]
        trainEd= dataLoaders[0][2]

        
        testIm=dataLoaders[1][0]
        testVe=dataLoaders[1][1]
        testEd= dataLoaders[1][2]

        # train loop

        for epoch in range(epochs):
            for batch1,batch2,batch3 in zip(trainIm,trainVe,trainEd):
                im,t = batch1
                v,_=batch2
                e,_ = batch3
                graph=(v,e)
                self.forward([im,graph])

        # test loop

        for epoch in range(epochs):
            for batch1,batch2,batch3 in zip(testIm,testVe,testEd):
                pass


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
        hiddenDim= 0
        # normLayer
        self.norm1 = nn.LayerNorm(hiddenDim)
        # mlp1 layer 
        self.mlp1 = nn.Sequential(
            nn.Linear(hiddenDim,hiddenDim),
            nn.GELU())
        # graph conv
        self.conv1 = GCNConv(in_channels, out_channels)
        #gelu
        self.gelu1 = nn.GELU()
        # graph conv
        self.conv2 = GCNConv(in_channels, out_channels)
        #gelu
        self.gelu2 = nn.GELU()
        # norm
        self.norm2 = nn.LayerNorm(hiddenDim)
        # mlp2
        self.mlp2 = nn.Sequential(
            nn.Linear(hiddenDim,hiddenDim),
            nn.GELU())
        # + residualmlp
        self.mlp3 = nn.Sequential(
            nn.Linear(hiddenDim,hiddenDim),
            nn.GELU())


    # def forward(self,data)
    def forward(self,data, edge_index):
        res = self.mlp3(data)

        x = self.norm1(data)
        x= self.mlp1(x)
        #x= self.conv1(x)
        x= self.conv1(x, edge_index)
        x= self.gelu1(x)
        #x= self.conv2(x)
        x= self.conv2(x, edge_index)
        x = self.gelu2(x)
        x = self.norm2(x)
        x = self.mlp2(x)

        x = res + x
        
        return x



class BertGraphEncoder(BertLayer):
    def __init__(self, config):
        super().__init__(config)
        self.gNN = GraphResidualBlock(config.hidden_size)

        for name, param in self.named_parameters():
            if not name.startswith("gNN"):
                param.requires_grad = False

    # edge index da controllare 
    def forward(self, hidden_states, attention_mask=None, head_mask=None,
                encoder_hidden_states=None, encoder_attention_mask=None,
                past_key_value=None, output_attentions=False, edge_index=None):
        
        self_attention_outputs = self.attention(hidden_states, attention_mask, head_mask, output_attentions, past_key_value,)
        

        attention_output = self_attention_outputs[0] # [B, L, D] output tensor after attention

        # GNN needs num_nodes, D
        B, L, D = attention_output.shape
        gnn_input = attention_output.reshape(B * L, D)
        gnn_output = self.gnn(gnn_input, edge_index)
       
        attention_output = self.gNN(attention_output)
        attention_output = gnn_output.view(B, L, D) # reshape for the rest of the pipeline

        # Apply the intermediate and output layers
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)

        return (layer_output,)
    

    
def main():

    resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
    modules = list(resnet.children())[:-1]
    featureExtractor = nn.Sequential(*modules)

    mT=model(backbone=featureExtractor)

    return