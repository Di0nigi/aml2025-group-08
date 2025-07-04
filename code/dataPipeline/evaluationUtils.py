from matplotlib import pyplot
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import torch

def lossAndAccGraph(trainLoss,trainAcc, valLos, valAcc):
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
    pyplot.show()
    return

def confusionMatAndFScores(targets,predictions,classes=[]):

    #print(targets)
    #print(predictions)

    targets = [t.argmax(dim=1) for t in targets]
    predictions = [ p.argmax(dim=1) for p in predictions]
    classes = [f"x" for x in range(12)]

    report = classification_report(targets, predictions, target_names=classes, digits=4)
    print(report)
    
    cm = confusion_matrix(targets, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')  
    pyplot.title("Confusion Matrix")
    pyplot.show()

    return