import torch.nn as nn
import torch
import dataPipeline.evaluationUtils as ev
from torchvision.models import resnet50, ResNet50_Weights 
from transformers import BertTokenizer, BertModel, BertConfig
from dataPipeline.embeddingUtils import embPipeline
from dataPipeline.dataSet import dataPipeline
from torch_geometric.nn import GCNConv 
from torch_geometric.data import Data,Batch
from transformers.models.bert.modeling_bert import BertLayer

class model(nn.Module):
    def __init__(self,backbone, dimEmbeddings=7080, gnn_dim=768, num_classes=12):
        super(model, self).__init__()
        self.loaded=False
        
        self.backBone = backbone

        #dimEmbeddings = 7080

        self.embed = nn.Sequential(
            nn.Linear(in_features=dimEmbeddings, out_features=gnn_dim),
            nn.ReLU(),
        )

        # Transformer encoder, inject BERT layers
        self.config = BertConfig.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        self.inChannels=self.config.hidden_size
        self.outChannels=self.config.hidden_size

        self.bert.encoder.layer = nn.ModuleList([
            BertGraphEncoder(self.config,self.inChannels,self.outChannels) for _ in range(self.config.num_hidden_layers)
        ])

        #for param in self.bert.parameters():
        #    param.requires_grad = False

        self.classifier = nn.Linear(gnn_dim, num_classes)
        
        return 
    
    def forward(self,data):
        im = data[0]

        graph = data[1]
        edges = data[1][1] 

        # Backbone pass, extract CNN features
        extractedFeatures = self.backBone(im)

        # Embeddings computations
        embeddings = embPipeline(extractedFeatures,graph)
        #print(type(embeddings))
        embeddings = torch.stack(embeddings)
        #print(embeddings.shape)
        embeddings=self.embed(embeddings)

        # bert with injected GNN

        #output = self.bert(inputs_embeds=embeddings, attention_mask=None, edge_index=edges)

        #output = embeddings.unsqueeze(0)#.permute(0,2,1)   
        output = embeddings
        # Pass through each transformer layer, injecting edge_index into each
        for layer_module in self.bert.encoder.layer:
            layer_outputs = layer_module(hidden_states=output, edge_index=edges)
            output = layer_outputs[0]  

        # CLS token
        cls_output = output[:, 0, :]
        logits = self.classifier(cls_output)

        return logits
    
    def trainL(self,dataLoaders,lossFunc, optimizer ,epochs=10):

        self.loss =lossFunc
        self.optim =optimizer

        trainIm=dataLoaders[0][0]
        trainVe=dataLoaders[0][1]
        trainEd= dataLoaders[0][2]

        
        testIm=dataLoaders[1][0]
        testVe=dataLoaders[1][1]
        testEd= dataLoaders[1][2]

        trainLossList, trainAccList = [], []
        valLossList, valAccList = [], []

        # train loop
        
        for epoch in range(epochs):
            print(f"Epoch [{epoch+1}/{epochs}]")
            self.train()
            totalLoss= 0.0
            correctT = 0
            totalT = 0
            for batch1,batch2,batch3 in zip(trainIm,trainVe,trainEd):
                im,t = batch1
                v,_=batch2
                e,_ = batch3
                graph=(v,e)
                y = self.forward([im,graph])
                t = t.view_as(y)
                #print(f"shape t {y.shape}") 
                #print(f"shape t {t.shape}") 
                loss = self.loss(y,t)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                #print(loss.item())
                totalLoss += loss.item()
                #print(f"Train loss: {loss.item():.4f}")

                preds = y.argmax(dim=1)
                correctT += (preds == t).sum().item()
                totalT += t.numel()
            trainAccuracy = correctT / max(totalT, 1)
            totalLoss= totalLoss/totalT
            trainLossList.append(totalLoss)
            trainAccList.append(trainAccuracy)


            print(f"Train Loss: {totalLoss:.4f}, Train Accuracy: {trainAccuracy*100:.4f}%")

            # test loop
            self.eval()

            totalEvalLoss = 0.0
            correct = 0
            total = 0
            targetList=[]
            predList=[]

            with torch.no_grad():
                for batch1,batch2,batch3 in zip(testIm,testVe,testEd):

                    im, t = batch1
                    v, _ = batch2
                    e, _ = batch3

                    graph = (v, e)

                    y = self.forward([im, graph])
                    t = t.view_as(y) 
                    loss = self.loss(y, t)
                    totalEvalLoss += loss.item()

                    # Classification: accuracy
                    predList.append(y)
                    targetList.append(t)
                    
                    preds = y.argmax(dim=1)
                    
                    correct += (preds == t).sum().item()
                    total += t.numel()

            avgEvalLoss = totalEvalLoss / max(len(testIm), 1)
            accuracy = correct / max(total, 1)
            valLossList.append(avgEvalLoss)
            valAccList.append(accuracy)

            print(f"Val Loss: {avgEvalLoss:.4f}, Val Accuracy: {accuracy*100:.4f}%\n")

        return  [trainLossList, trainAccList , valLossList, valAccList], [predList, targetList] 
    
    def predict(self):

        return
    
    def save(self,path):
        torch.save(self.state_dict(), path)
        return path

    def load(self,path,location=None):
        self.load_state_dict(torch.load(path, map_location=location))
        self.loaded=True
        return
    
class GraphResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hiddenDim):
        super().__init__()
        # hiddenDim= 0
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
    def __init__(self, config,inChannels,outChannels):
        super().__init__(config)
        self.gNN = GraphResidualBlock(in_channels=inChannels,out_channels=outChannels,hiddenDim=config.hidden_size)

        for name, param in self.named_parameters():
            if not name.startswith("gNN"):
                param.requires_grad = False

    def forward(self, hidden_states, attention_mask=None, head_mask=None,
                encoder_hidden_states=None, encoder_attention_mask=None,
                past_key_value=None, output_attentions=False, edge_index=None):
        
        #print("hiddenst")
        #print(hidden_states.shape)
        #print(edge_index.shape)
        #edges= edge_index.squeeze().permute(1,0)
        #print(getMaxIndex(edges))
        #print(edges)
        #if hidden_states.is_bool():
        #hidden_states = hidden_states.float()


        
        self_attention_outputs = self.attention(hidden_states)
        

        attention_output = self_attention_outputs[0] # [B, L, D] output tensor after attention
        #print("eo")
        #print(attention_output.shape)
        # GNN needs num_nodes, D
        B, L, D = attention_output.shape

        graphBatched = Batch.from_data_list([Data(x=attention_output[i], edge_index=edge_index[i].t().contiguous()) for i in range(edge_index.size(0))])

        #gnn_input = attention_output.reshape(B * L, D)
        
        gnn_output = self.gNN(graphBatched.x, graphBatched.edge_index)

        #print("e")
       
        #attention_output = self.gNN(attention_output,edge_index)
        gnn_output = gnn_output.view(B, L, D) # reshape for the rest of the pipeline
        #print(gnn_output.shape)

        # Apply the intermediate and output layers
        intermediate_output = self.intermediate(gnn_output)
        layer_output = self.output(intermediate_output, gnn_output)

        return (layer_output,)
       
def main():

    data = dataPipeline("D:\dionigi\Documents\Python scripts\\aml2025Data\dataNorm",split=0.8,batches=1,classes=12)
   

    resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
    modules = list(resnet.children())[:-1]
    featureExtractor = nn.Sequential(*modules)

    # Input shape: whatever
    # Output shape: torch.Size([1, 2048, 1, 1])


    mT=model(backbone=featureExtractor)

    lossFunction = torch.nn.L1Loss()
    optimizer = torch.optim.AdamW(mT.parameters(), lr=1e-4, weight_decay=1e-5)

    [tL, tAcc, teL, teAcc], [pred, targs] =mT.trainL(dataLoaders=data,lossFunc=lossFunction,optimizer=optimizer,epochs=1)

    #ev.lossAndAccGraph(tL, tAcc, teL, teAcc)
    ev.confusionMatAndFScores(pred,targs)

    return "done"

if __name__ == "__main__":   
    print(main())
