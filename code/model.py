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
    def __init__(self,backbone,device, dimEmbeddings=7080, gnn_dim=768, num_classes=12):
        super(model, self).__init__()
        self.loaded=False
        
        self.backBone = backbone

        self.dev = device

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
            BertGraphEncoder(self.config,self.inChannels,self.outChannels,device=self.dev) for _ in range(self.config.num_hidden_layers)
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
        graphBatched = graphBatched.to( self.dev)
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
    #device = 0
    if torch.cuda.is_available():
        print("CUDA enabled")
        device = torch.device("cuda")
    else:
        print("CUDA not available")
        device = torch.device("cpu")

    data = dataPipeline("D:\dionigi\Documents\Python scripts\\aml2025Data\dataNormL",split=0.8,batches=16,classes=12)
   

    resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
    modules = list(resnet.children())[:-1]
    featureExtractor = nn.Sequential(*modules)

    # Input shape: whatever
    # Output shape: torch.Size([1, 2048, 1, 1])


    mT=model(backbone=featureExtractor,device=device)
    mT.to(device)

    lossFunction = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(mT.parameters(), lr=1e-4, weight_decay=1e-5)

    [tL, tAcc, teL, teAcc], [pred, targs] =mT.trainL(dataLoaders=data,lossFunc=lossFunction,optimizer=optimizer,epochs=1)

    #print([pred, targs])

    '''[pred, targs] = [[torch.tensor([[-0.1821, -1.1024, -1.6555,  0.4486,  0.7422, -0.2335,  0.9218,  0.3999,
          0.2800, -0.4259,  0.2247,  1.4254],
        [-0.1757, -1.0360, -1.6373,  0.4360,  0.7465, -0.2144,  1.0030,  0.3926,
          0.1779, -0.4068,  0.1847,  1.4592],
        [-0.1513, -1.0966, -1.6294,  0.3786,  0.6989, -0.2010,  0.9874,  0.3709,
          0.2586, -0.4279,  0.1575,  1.4740],
        [-0.2235, -1.1038, -1.6960,  0.4443,  0.7384, -0.2643,  0.9733,  0.3997,
          0.1987, -0.4399,  0.1530,  1.4961],
        [-0.1866, -1.0786, -1.6476,  0.4639,  0.7343, -0.2489,  0.9460,  0.4055,
          0.2628, -0.4152,  0.2202,  1.4546],
        [-0.1746, -1.0417, -1.6206,  0.4772,  0.7441, -0.2190,  1.0052,  0.3992,
          0.1781, -0.4368,  0.1699,  1.4816],
        [-0.1832, -1.0561, -1.6461,  0.4645,  0.7519, -0.2365,  0.9691,  0.3889,
          0.2778, -0.4212,  0.2180,  1.4513],
        [-0.1060, -1.0424, -1.5828,  0.4698,  0.7148, -0.2190,  0.9975,  0.4262,
          0.1751, -0.4002,  0.1729,  1.4916],
        [-0.1625, -1.0478, -1.6198,  0.4628,  0.7723, -0.2360,  0.9582,  0.3823,
          0.2498, -0.4422,  0.1876,  1.4649],
        [-0.1792, -1.0582, -1.6481,  0.4633,  0.7405, -0.2405,  0.9493,  0.3858,
          0.2467, -0.4235,  0.2083,  1.4635],
        [-0.1540, -1.0685, -1.6376,  0.4620,  0.7089, -0.2464,  1.0082,  0.3871,
          0.2019, -0.4184,  0.1445,  1.5237],
        [-0.2133, -1.0994, -1.6922,  0.4259,  0.7632, -0.2535,  0.9439,  0.3855,
          0.2340, -0.4258,  0.1990,  1.4456],
        [-0.1621, -1.0523, -1.6415,  0.4519,  0.7103, -0.2330,  0.9519,  0.3899,
          0.2355, -0.3925,  0.2151,  1.4677],
        [-0.1658, -1.0885, -1.6336,  0.4583,  0.7023, -0.2367,  0.9445,  0.3688,
          0.2440, -0.4405,  0.1770,  1.4558],
        [-0.1234, -1.0634, -1.6564,  0.5083,  0.6712, -0.2366,  0.9906,  0.3639,
          0.1722, -0.3892,  0.1451,  1.5248],
        [-0.1377, -1.0776, -1.6195,  0.4535,  0.6718, -0.2500,  0.9772,  0.3778,
          0.1834, -0.4087,  0.1793,  1.4947]], device='cuda:0'), torch.tensor([[-0.1594, -1.0590, -1.6460,  0.4551,  0.7268, -0.2557,  1.0176,  0.3886,
          0.2143, -0.4125,  0.1576,  1.5054],
        [-0.1716, -1.0657, -1.6464,  0.4715,  0.7422, -0.2549,  0.9945,  0.4085,
          0.1934, -0.4328,  0.1586,  1.5050],
        [-0.1695, -1.0916, -1.6675,  0.4612,  0.7286, -0.2388,  0.9831,  0.3889,
          0.2461, -0.4382,  0.1740,  1.4728],
        [-0.1701, -1.0763, -1.6442,  0.4565,  0.6831, -0.2633,  0.9850,  0.3745,
          0.2161, -0.4022,  0.1569,  1.5314],
        [-0.1507, -1.0540, -1.6242,  0.4493,  0.7016, -0.2404,  0.9773,  0.3717,
          0.2127, -0.4192,  0.1669,  1.4804],
        [-0.1600, -1.0547, -1.6317,  0.4844,  0.7723, -0.2397,  0.9513,  0.4043,
          0.2647, -0.4395,  0.1849,  1.4626],
        [-0.1116, -1.0518, -1.6072,  0.4768,  0.6741, -0.2301,  0.9745,  0.3583,
          0.1616, -0.4133,  0.1397,  1.5406],
        [-0.1740, -1.0851, -1.6321,  0.4163,  0.7000, -0.2626,  0.9474,  0.3860,
          0.2337, -0.4350,  0.1805,  1.4771]], device='cuda:0')], [torch.tensor([ 8,  4,  1,  3,  5, 10,  2,  9,  5,  2,  0,  7,  2,  3,  6,  6],
       device='cuda:0'), torch.tensor([ 0, 10,  5,  1,  9,  3, 10,  7], device='cuda:0')]]'''

   # print(len(targs))
   # print(len(pred))
    #ev.lossAndAccGraph(tL, tAcc, teL, teAcc)
    #ev.confusionMatAndFScores(targs,pred)

    return "done"

if __name__ == "__main__":   
    print(main())
