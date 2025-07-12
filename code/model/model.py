import torch.nn as nn
import torch
import evaluation.evaluationUtils as ev
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights, resnet34, ResNet34_Weights, mobilenet_v2, MobileNet_V2_Weights
from transformers import BertTokenizer, BertModel, BertConfig
from dataPipeline.embeddingUtils import embPipeline,addClsTkCons
from dataPipeline.dataSet import dataPipeline
from torch_geometric.nn import GCNConv 
from torch_geometric.data import Data,Batch
from transformers.models.bert.modeling_bert import BertLayer
import gc

class model(nn.Module):
    def __init__(self,backbone,device,numLayers=12, dimEmbeddings=5544, gnn_dim=768,dropout_rate=0.1, num_classes=9):

        #7080
        #5544
        #6312

        self.bestVal=0

        self.nL = numLayers

        super(model, self).__init__()
        self.loaded=False
        
        self.backBone = backbone


        self.pool =  nn.AdaptiveAvgPool2d((1, 1)) 

        self.dev = device

        #dimEmbeddings = 7080

        self.embed = nn.Sequential(
            nn.Linear(in_features=dimEmbeddings, out_features=gnn_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)  

        )

        # Transformer encoder, inject BERT layers
        self.config = BertConfig.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        self.inChannels=self.config.hidden_size
        self.outChannels=self.config.hidden_size

        self.bert.encoder.layer = nn.ModuleList([
            BertGraphEncoder(self.config,self.inChannels,self.outChannels,device=self.dev) for _ in range(self.nL)#(self.config.num_hidden_layers)
        ])

        #for param in self.bert.parameters():
        #    param.requires_grad = False

        self.dropout=nn.Dropout(dropout_rate)

        self.classifier = nn.Linear(gnn_dim, num_classes)

        self.cls = nn.Parameter(torch.zeros(size=(1,dimEmbeddings)))
        nn.init.trunc_normal_(self.cls, std=0.02)
        
        return 
    
    def forward(self,data):
        im = data[0]

        graph = data[1]
        edges = data[1][1] 
        #print(edges.shape)
        edges=addClsTkCons(edges)
        #print(edges.shape)

        # Backbone pass, extract CNN features
        extractedFeatures = self.backBone(im)
        #extractedFeatures = self.pool(extractedFeatures)
        #extractedFeatures = extractedFeatures.view(extractedFeatures.size(0), -1)
        #print(extractedFeatures.shape)

        # Embeddings computations

        self.cls=self.cls.to(self.dev)
        
        embeddings = embPipeline(extractedFeatures,graph,self.cls)
        
        #print(type(embeddings))
        embeddings = torch.stack(embeddings)
        #print(embeddings.shape)
        
        embeddings=self.embed(embeddings)
        #print(embeddings.shape)

        # bert with injected GNN

        #output = self.bert(inputs_embeds=embeddings, attention_mask=None, edge_index=edges)

        #output = embeddings.unsqueeze(0)#.permute(0,2,1)
        output1 = embeddings   
        output = embeddings
        # Pass through each transformer layer, injecting edge_index into each
        for layer_module in self.bert.encoder.layer:
            layer_outputs = layer_module(hidden_states=output, edge_index=edges)
            output = layer_outputs[0]  

        output += output1
        
        #print(output.shape)

        # CLS token
        cls_output = output[:, 0, :]
        #cls_output=output

        cls_output=self.dropout(cls_output)


        logits = self.classifier(cls_output)

        #print("forward")


        return logits
    
    def trainL(self,dataLoaders,lossFunc, optimizer ,epochs=10,patience=5):
        targetList=[]
        predList=[]

        self.loss =lossFunc
        self.optim =optimizer

        #self.load(path="D:\dionigi\Documents\Python scripts\\aml2025Data\models\\bestModel.pth")

        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_model_state = None

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

                im = im.to(self.dev)
                t = t.to(self.dev)
                v = v.to(self.dev)
                e = e.to(self.dev)

                graph=(v,e)
                y = self.forward([im,graph])
                
                t = t.squeeze(1)
                t = t.argmax(dim=1).long() 
                #print(t.shape)
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

                    im = im.to(self.dev)
                    t = t.to(self.dev)
                    v = v.to(self.dev)
                    e = e.to(self.dev)

                    graph = (v, e)

                    y = self.forward([im, graph])
                    #t = t.view_as(y) 
                    #t = t.view(-1).long()
                    t = t.squeeze(1)
                    t = t.argmax(dim=1).long() 
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

            if avgEvalLoss < best_val_loss:
                best_val_loss = avgEvalLoss
                epochs_no_improve = 0
                best_model_state = self.state_dict()
                self.bestVal = accuracy*100  
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch+1} due to no improvement in validation loss.")
                    self.load_state_dict(best_model_state)  
                    break
        #self.bestVal = best_val_loss
        return  [trainLossList, trainAccList , valLossList, valAccList], [predList, targetList] 
    
    def predict(self,data):
        d1,d2,d3 = data
        
        ret1 = []
        ret2 = []
        correct =0
        total =0
        self.eval()
        with torch.no_grad():
            for batch1,batch2,batch3 in zip(d1,d2,d3):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                im, t = batch1
                v, _ = batch2
                e, _ = batch3

                im = im.to(self.dev)
                t = t.to(self.dev)
                v = v.to(self.dev)
                e = e.to(self.dev)

                graph = (v, e)

                y = self.forward([im, graph])
                t = t.squeeze(1)
                t = t.argmax(dim=1).long() 
                ret1.append(y)
                ret2.append(t)
                preds = y.argmax(dim=1)        
                correct += (preds == t).sum().item()
                total += t.numel()
        acc = correct/total
        
        return acc, ret1, ret2
    
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
    def __init__(self, config,inChannels,outChannels,device):
        super().__init__(config)
        self.dev= device
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
        graphBatched = graphBatched.to(self.dev)
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
    
def weightDataSet(dataSet):
    train = dataSet[0][0]
    

    labelList = []

    for b in train:
        _,label = b
        #print(label.shape)
        label = int(torch.argmax(label))
        labelList.append(label)

    labelsTensor = torch.tensor(labelList)
    classCounts = torch.bincount(labelsTensor)

    # Compute weights: inverse frequency
    weights = 1.0 / classCounts.float()
    weights = weights / weights.sum()  
    return weights

       
def main():
    #device = 0
    if torch.cuda.is_available():
        print("CUDA enabled")
        device = torch.device("cuda")
        print(torch.cuda.get_device_name(device))
    else:
        print("CUDA not available")
        device = torch.device("cpu")
    
    #device = torch.device("cpu")


    data = dataPipeline("D:\dionigi\Documents\Python scripts\\aml2025Data\dataNorm",split=0.8,batches=16,classes=9)



    ev.classesDistribution(data)

    classWeights = weightDataSet(data)

    #resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
    resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
    modules = list(resnet.children())[:-1]
    featureExtractor = nn.Sequential(*modules)

    # Input shape: whatever
    # Output shape: torch.Size([1, 2048, 1, 1])

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    
    mT=model(backbone=featureExtractor,device=device,numLayers=2,dropout_rate=0.5,num_classes=5)
    mT.to(device)

    
    lossFunction = torch.nn.CrossEntropyLoss(weight=classWeights.to(device))
    optimizer = torch.optim.AdamW(mT.parameters(), lr=1e-05, weight_decay= 1e-05)
    #for t in range(3):
    #mT.load(path="D:\dionigi\Documents\Python scripts\\aml2025Data\models\\bestModel.pth")
    restot=[]
    #for _ in range(2):
    res =mT.trainL(dataLoaders=data,lossFunc=lossFunction,optimizer=optimizer,epochs=100,patience=5)
#    restot.append(res)
    # [trainLossList, trainAccList , valLossList, valAccList], [predList, targetList] 
    #resT=[[restot[0][0][0]+restot[1][0][0],restot[0][0][1]+restot[1][0][1],restot[0][0][2]+restot[1][0][2],restot[0][0][3]+restot[1][0][3]],[ restot[0][1][0]+ restot[1][1][0], restot[0][1][1]+ restot[1][1][1] ]]
    mT.save(path="D:\dionigi\Documents\Python scripts\\aml2025Data\models\\bestModelD.pth")
    print(mT.bestVal)

    #mT.load(path="D:\dionigi\Documents\Python scripts\\aml2025Data\models\\5model95.pth")
    #a,p,t =mT.predict(data[0])
    #print(a)

    #print([pred, targs])
    #[tL, tAcc, teL, teAcc], [pred, targs] = res

    #print(res)



    #print(len(targs))
    #print(len(pred))
    
    #ev.lossAndAccGraph(tL, tAcc, teL, teAcc)

    #ev.confusionMatAndFScores(t,p)

    ev.displayData(res)

    #del mT
    #torch.cuda.empty_cache()
    #gc.collect()

    return "done"

if __name__ == "__main__":   
    print(main())
