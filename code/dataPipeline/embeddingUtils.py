#import torch.nn as nn
import torch
#from torchvision import models
import os
import json
import numpy as np


def loadGraphs(dir):
    d=[]
    for subDir in os.listdir(dir):
        n = f"{dir}\{subDir}"
        if "_lnd" in subDir:
            sd=[]
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
                        break
                    else:
                        e=data.pop()
                        e = [[e[x]['start'],e[x]['end']] for x in range(len(e))]
                        v = [[data[x]['x'],data[x]['y'],data[x]['z'],data[x]['visibility']] for x in range(len(data))]
                        sd.append((torch.tensor(v),torch.tensor(e)))
            d.append(sd)
    return d

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

def concatCoor(verteces):

    v = np.array(verteces)
    #print(v.shape)
    v= v.flatten()
    #print(v.shape)

    return torch.tensor(v)

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
        emb= torch.concat((elem,concatCoor(vt),buildAdjMat(ed)))
        #print(emb.shape)
        embeddings.append(emb)
    return embeddings


#def saveData(data):
#    return





def main():
    d=loadGraphs("D:\dionigi\Documents\Python scripts\\aml2025Data\dataNorm")
    #print(d[0][1])
    #print(buildAdjMat(d[0][1]))
    #print(concatCoor(d[0][0]))
    return

if __name__ == "__main__":   
    main()