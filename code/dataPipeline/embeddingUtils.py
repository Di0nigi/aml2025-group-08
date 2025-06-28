#import torch.nn as nn
#import torch
#from torchvision import models
import os
import json


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
     
                    v=data.pop()
            d.append(sd)




def embPipeline(data):
    return


def saveData(data):
    return





def main():
    loadGraphs("D:\dionigi\Documents\Python scripts\\aml2025Data\dataNorm")
    return

main()