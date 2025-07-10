from matplotlib import pyplot
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import torch
from torch import tensor

def lossAndAccGraphOld(trainLoss,trainAcc, valLos, valAcc,show=True):
    pyplot.figure(figsize=(10,3))
    ax = pyplot.subplot(121)
    # plot training and validation loss values of CV network over epochs
    ax.plot(range(len(trainLoss)), trainLoss, label='Training Loss')
    ax.plot(range(len(valLos)), valLos, label='Validation Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()


    ax = pyplot.subplot(122)
    # plot training and validation accuracy values of CV network over epochs
    ax.plot(range(len(trainAcc)), trainAcc, label='Training Accuracy')
    ax.plot(range(len(valAcc)), valAcc, label='Validation Accuracy')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.legend()

    #pyplot.title("Loss & Accuracy")
    pyplot.tight_layout() 
    if show:
        pass
        #pyplot.show()

    return 

def lossAndAccGraph(trainLoss, trainAcc, valLoss, valAcc,show=True):

    fig, axs = pyplot.subplots(1, 2, figsize=(10, 4))

    axs[0].plot(range(len(trainLoss)), trainLoss, label='Training Loss')
    axs[0].plot(range(len(valLoss)), valLoss, label='Validation Loss')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].set_title("Loss")
    axs[0].legend()

    axs[1].plot(range(len(trainAcc)), trainAcc, label='Training Accuracy')
    axs[1].plot(range(len(valAcc)), valAcc, label='Validation Accuracy')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Accuracy')
    axs[1].set_title("Accuracy")
    axs[1].legend()

    pyplot.tight_layout() 


    if show:
        
        pyplot.show()

    return fig

def confusionMatAndFScores(targets,predictions,classes="D:\dionigi\Documents\Python scripts\\aml2025-group-08\docs\classes5.txt",show=True):

    targets = [t.detach().cpu() for t in targets]
    predictions = [ p.detach().cpu().argmax(dim=1) for p in predictions]
    targets =torch.concatenate(targets)
    predictions = torch.concatenate(predictions)

    classList = []
    with open (classes, mode="r",encoding="utf-8") as f:
        for lines in f:
            c=lines.replace("\n",'')
            classList.append(f"{c[:3]}.")


    report = classification_report(targets, predictions, target_names=classList, digits=4,zero_division=0)
    
    print(f"{''.join(['-' for x in range(100)])}")
    print(report)
    
    cm = confusion_matrix(targets, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=classList)
    if show:
        disp.plot()  
        pyplot.title("Confusion Matrix")
        pyplot.show()

        return 
    
    else:

        fig, ax = pyplot.subplots(figsize=(6, 5))

        disp.plot(ax=ax, xticks_rotation=45, colorbar=True)
        
        ax.set_title("Confusion Matrix")

        return fig

def classesDistribution(dataSet,classes="D:\dionigi\Documents\Python scripts\\aml2025-group-08\docs\classes5.txt"):
    
    train = dataSet[0][0]
    test=dataSet[1][0]

    labels = []

    for b in train:
      _, l = b
      l = l.squeeze(1).argmax(dim=1).long()
      labels.append(l)

    for b in test:
        _, l2 = b
        l2 = l2.squeeze(1).argmax(dim=1).long()
        labels.append(l2)
        
        allLabels = torch.cat(labels)
    
    print(len(allLabels))


   
    uniques, counts = torch.unique(allLabels, return_counts=True)

    
    uniques = uniques.cpu().numpy()
    counts = counts.cpu().numpy()

    
    #class_names = [str(i) for i in unique_labels]
    classList = []
    with open (classes, mode="r",encoding="utf-8") as f:
        for lines in f:
            c=lines.replace("\n",'')
            classList.append(f"{c[:3]}.")

    
    pyplot.figure(figsize=(10, 5))
    pyplot.bar(classList, counts, color='skyblue')
    pyplot.xlabel('Class')
    pyplot.ylabel('Number of Samples')
    pyplot.title('Class Distribution in Dataset')
    pyplot.xticks(rotation=45)
    pyplot.tight_layout()
    pyplot.show()


    return
 


def displayData(data):

    [tL, tAcc, teL, teAcc], [pred, targs] = data
  
    lossAccFig = lossAndAccGraph(tL, tAcc, teL, teAcc,show=False)
    confFig = confusionMatAndFScores(targs, pred,show=False)

    #lossAccFig.show()
    #confFig.show()

    pyplot.show()

    return



def main ():
    test = ([[0.183706323724044], [0.07368421052631578], [2.876679301261902], [0.08333333333333333]], [[tensor([[-0.0877,  1.3746, -1.2479,  0.8902,  0.6213, -0.6913,  0.8800,  0.4620,
          0.0183, -0.8201,  0.7164, -1.1580],
        [-0.1053,  1.4161, -1.1753,  0.8496,  0.5638, -0.6421,  0.8070,  0.4516,
          0.0338, -0.7907,  0.7094, -1.1592],
        [-0.0342,  1.3926, -1.2158,  0.8723,  0.5987, -0.6166,  0.8496,  0.4557,
          0.0910, -0.7914,  0.7008, -1.1965],
        [-0.0654,  1.3812, -1.1864,  0.8497,  0.5976, -0.6398,  0.8546,  0.4539,
          0.0649, -0.7989,  0.6951, -1.1328],
        [-0.1340,  1.3965, -1.1848,  0.8601,  0.6274, -0.6708,  0.8241,  0.4660,
          0.0341, -0.8166,  0.7067, -1.1148],
        [-0.1174,  1.3643, -1.1802,  0.8601,  0.6447, -0.6859,  0.7878,  0.5029,
          0.0599, -0.8547,  0.6892, -1.1273],
        [-0.0411,  1.3861, -1.2981,  0.8740,  0.6498, -0.6508,  0.8581,  0.4414,
          0.0014, -0.8063,  0.6912, -1.2817],
        [-0.0833,  1.3702, -1.1829,  0.8476,  0.6226, -0.6822,  0.8299,  0.4674,
          0.0392, -0.7973,  0.7160, -1.1268],
        [-0.0651,  1.3701, -1.2118,  0.8682,  0.6116, -0.6620,  0.8546,  0.4619,
          0.0645, -0.7847,  0.6574, -1.1715],
        [-0.0685,  1.3717, -1.2262,  0.8685,  0.6261, -0.6523,  0.8700,  0.4516,
          0.0788, -0.8122,  0.6746, -1.1993],
        [-0.0532,  1.3554, -1.2617,  0.8861,  0.6589, -0.6407,  0.9128,  0.4641,
          0.0855, -0.8124,  0.6957, -1.1837],
        [-0.0607,  1.4200, -1.2530,  0.8835,  0.6478, -0.6381,  0.8115,  0.4517,
          0.0477, -0.7967,  0.6901, -1.2100],
        [-0.0635,  1.3785, -1.1894,  0.8763,  0.5791, -0.6444,  0.8263,  0.4615,
          0.0938, -0.8115,  0.6693, -1.1975],
        [-0.1005,  1.3566, -1.1579,  0.8726,  0.5609, -0.6761,  0.8324,  0.4672,
          0.0846, -0.8272,  0.6807, -1.1525],
        [-0.0811,  1.3779, -1.2139,  0.8736,  0.5992, -0.6704,  0.8479,  0.4582,
          0.0463, -0.8254,  0.6798, -1.1833],
        [-0.1002,  1.3662, -1.1853,  0.8676,  0.6214, -0.6696,  0.8299,  0.4882,
          0.0626, -0.8158,  0.7221, -1.1341]], device='cuda:0'), tensor([[-6.9445e-02,  1.3743e+00, -1.2236e+00,  8.7031e-01,  5.4456e-01,
         -6.9648e-01,  8.9012e-01,  4.6693e-01,  5.1813e-02, -8.0776e-01,
          7.6005e-01, -1.2052e+00],
        [-8.5654e-02,  1.3452e+00, -1.1871e+00,  8.5288e-01,  5.7975e-01,
         -6.8177e-01,  8.5891e-01,  4.5765e-01,  3.4722e-02, -7.7407e-01,
          6.8458e-01, -1.1517e+00],
        [-1.2307e-01,  1.3689e+00, -1.1102e+00,  8.7421e-01,  5.8863e-01,
         -6.5635e-01,  8.1078e-01,  4.6268e-01,  7.0881e-02, -7.9403e-01,
          6.9086e-01, -1.1163e+00],
        [-1.2874e-01,  1.3893e+00, -1.2103e+00,  8.5533e-01,  5.8757e-01,
         -6.6587e-01,  8.3494e-01,  4.2763e-01, -9.8991e-04, -7.7985e-01,
          7.1457e-01, -1.1532e+00],
        [-4.9922e-02,  1.3631e+00, -1.2527e+00,  8.6957e-01,  6.0899e-01,
         -6.8049e-01,  8.9276e-01,  4.4763e-01,  7.9788e-02, -8.1683e-01,
          7.1587e-01, -1.1871e+00],
        [-6.7856e-02,  1.3646e+00, -1.2270e+00,  8.8755e-01,  5.9767e-01,
         -6.8684e-01,  8.9983e-01,  4.4812e-01,  5.2324e-02, -8.0333e-01,
          7.0723e-01, -1.1680e+00],
        [-9.8870e-02,  1.3652e+00, -1.1424e+00,  8.5694e-01,  5.8466e-01,
         -6.7003e-01,  8.3266e-01,  4.5054e-01,  4.8850e-02, -7.6514e-01,
          6.8462e-01, -1.1621e+00],
        [-5.2888e-02,  1.3882e+00, -1.2327e+00,  8.5247e-01,  6.2639e-01,
         -6.3714e-01,  8.2505e-01,  4.1658e-01,  2.6572e-02, -7.9789e-01,
          7.0548e-01, -1.2453e+00]], device='cuda:0')], [tensor([10,  2, 10, 11,  8,  3,  0,  7,  9, 11, 11,  4,  0,  9,  1,  8],
       device='cuda:0'), tensor([2, 7, 7, 2, 6, 5, 4, 1], device='cuda:0')]])
    
    displayData(test)

    return

if __name__ == "__main__":   
    main()