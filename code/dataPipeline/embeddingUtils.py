#import torch.nn as nn
import torch
#from torchvision import models
import os
import json
import numpy as np


def loadGraphs(dir):
    d=[]
    im=[]
    for subDir in os.listdir(dir):
        n = f"{dir}\{subDir}"
        if "_lnd" in subDir:
            sd=[]
            si=[]
            for file in os.listdir(n):
                with open(f"{n}\\{file}", 'r') as f:
                    data = json.load(f)['keypoints']
                    #print(len(data))
                    if len(data)==0:
                        f.close()
                        print(f"{n}\\{file}")
                        os.remove(f"{n}\\{file}")
                        print(f"{n[:-4]}\\{file[:-9]}.png")
                        os.remove(f"{n[:-4]}\\{file[:-9]}.png")
                        #print(data)
                    else:
                        e=data.pop()
                        e = [[e[x]['start'],e[x]['end']] for x in range(len(e))]
                        v = [[data[x]['x'],data[x]['y'],data[x]['z'],data[x]['visibility']] for x in range(len(data))]
                        sd.append((torch.tensor(v),torch.tensor(e)))
                        si.append(f"{n[:-4]}\\{file[:-9]}.png")
            d.append(sd)
            im.append(si)
    return d,im

def buildAdjMat(edges):
    #[[2,3],[3,4],[5,6]]
    
    nodes = sorted(set([node for edge in edges for node in edge]))

   
    nodeToIdx = {node.item(): idx for idx, node in enumerate(nodes)}

    
    N = len(nodes)
    adjMatrix = np.zeros((N, N), dtype=int)

    
    for edge in edges:
        u, v = nodeToIdx[edge[0].item()], nodeToIdx[edge[1].item()]
        adjMatrix[u][v] = 1
        adjMatrix[v][u] = 1  

    adj = adjMatrix.flatten()

    return torch.tensor(adj)

def concatCoor(vertices):

    #verteces=verteces.detach().cpu()

    #v = np.array(verteces)
    #print(v.shape)
    #v= v.flatten()
    #print(v.shape)
    v = vertices.flatten()

    return v #torch.tensor(v) v

## takes matrix of tensors and matrix of tuples of matricies returns matrix of vectors

def embPipeline(data,graphs):
    embeddings=[]
    for ind, elem in enumerate(data):
        vt=graphs[0][ind]
        ed=graphs[1][ind]
        elem=elem.squeeze()
       # print(elem.shape)
       # print(concatCoor(vt).shape)
       # print(buildAdjMat(ed).shape)
        #mI=getMaxIndex(ed.squeeze().permute(1,0))
        #print(mI)
        mI= 32 # should be consistent across 
        adjMat = buildAdjMat(ed).to(elem.device)
        #print(elem.device)
        #print(concatCoor(vt).device)
        #print(buildAdjMat(ed).device)
        emb= torch.concat((elem,concatCoor(vt),adjMat))
        emb = torch.stack([emb for x in range(mI+1)])
       #print("shape")
        #print(emb.shape)
        embeddings.append(emb)
    return embeddings


#def saveData(data):
#    return

def getMaxIndex(edge_index: torch.Tensor):
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"Invalid edge_index shape: {edge_index.shape}, expected [2, num_edges]")
    
    return int(edge_index.max().item())




def main():
    #d=loadGraphs("D:\dionigi\Documents\Python scripts\\aml2025Data\dataNorm")
    #print(d[0][1])
    #print(buildAdjMat(d[0][1]))
    #print(concatCoor(d[0][0]))
    return

if __name__ == "__main__":   
    main()