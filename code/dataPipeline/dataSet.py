
import numpy  as np
import os
import torch 
from PIL import Image
import matplotlib.pyplot as plt
import mediapipe as mp
import cv2
import json

def loadData(path):
    r = os.listdir(path)
    out=[]
    l=[]
    for elem in r:
        n = os.path.join(path,elem)
        if os.path.isdir(n):
            l=[]
            for e in os.listdir(n):
                l.append(os.path.join(n,e))
            #print(len(l))
        out.append(l)
    return out

'''
def countDims(l):
    out =[]
    for lis in l:
        for elem in lis: 
            im = Image.open(elem)
            im=np.array(im)
            out.append(im.shape)
    return out

def plotDims(lis):
    lis.sort()
    x1 = [x[0]+x[1] for x in lis]
    #y1 = [y.count(y[0]) for y in lis]
    y0=[y for y in range(len(lis))]

    x2 = [x[1] for x in lis ]
    #y2 = [y.count(y[1]) for y in lis]

    fig,axes = plt.subplots(2,1)

    #axes = np.reshape(axes,newshape=(len(lis),))

    axes[0].bar(y0,x1)
    axes[1].bar(y0,x2)

    plt.tight_layout()
    plt.show()

    return

def applyCrop(im,dim,sz):
   # print("cShape")
    #print(im.shape)
    if (im.shape[dim] == sz):
        return im
    elif dim:
        _, width, _ = im.shape
        start = (width - sz) // 2
        end = start + sz
        outIm=im[:, start:end, :]
    else:
        height, _, _ = im.shape
        start = (height - sz) // 2
        end = start + sz
        outIm=im[start:end, :, :]
    return outIm

def applyPadding(im,dim,sz):
    padCol=np.array([255,255,255])
    if (im.shape[dim] == sz):
        return im
    elif dim:
        n=sz-im.shape[1]
        lf = n//2
        if (lf<=0):
            lf=1
        rt = n-lf
        if (rt<=0):
            rt=1
        padl = np.array([[padCol for x in range(lf)] for y in range(im.shape[0])])
        padr = np.array([[padCol for x in range(rt)]for y in range(im.shape[0])])
        #padl = np.full((im.shape[0], lf), (255,255,255))
        #padr = np.full((im.shape[0], rt),  (255,255,255))
       # print("s")
        #print(lf)
        #print(rt)
        #print(im.shape)
        #print("e")
        outIm = np.hstack((padl, im, padr))
    else:
        n=sz-im.shape[0]
        lf = n//2
        if (lf<=0):
            lf=1
        rt = n-lf
        if (rt<=0):
            rt=1
        padl = np.array([[padCol for x in range(im.shape[1])] for y in range(lf)])
        padr = np.array([[padCol for x in range(im.shape[1])]for y in range(rt)])
       # print("s")
        #print(padl.shape)
        #print(padr.shape)
        #print(im.shape)
        #print("e")
        outIm = np.concatenate((padl, im, padr))
    
    return outIm

'''

def normDataset(data,targetShape=(500,500,3)):
    out=[]
    out2=[]
    for dir in data:
        #dirs = os.listdir(dir)
        #if ()
        d=[]
        d2=[]
        for pic in dir:
            im = Image.open(pic)
            im=np.array(im)
            dim = im.shape
            if len(dim)==2:
                im = np.stack([im] * 3, axis=-1)
            if len(dim)==3 and im.shape[2]>3:
                im=im[:, :, :3]
            '''if dim == targetShape:
                break
            if dim[0]>targetShape[0]:
                im = applyCrop(im,0,targetShape[0])
            else:
                im = applyPadding(im,0,targetShape[0])
            if dim[1]>targetShape[1]:
                im = applyCrop(im,1,targetShape[1])
            else:
                im = applyPadding(im,1,targetShape[1])
            #print(im.shape)'''
            im = im.astype(np.uint8)
            lndMarks = getLandMarks(im)
            im = cv2.resize(im,(targetShape[0],targetShape[1]))
            d.append(im)
            d2.append(lndMarks)
        out.append(d)
        out2.append(d2)


    return out,out2

def saveData(lis,lis2,directory):
    d=os.listdir(directory)
    
    #print(d)
    for dir in range(len(lis)):
        if str(dir) not in d:
            os.mkdir(os.path.join(directory,str(dir)))
        for n,im in enumerate(lis[dir]):
            #print(im.shape)
            pic=Image.fromarray(im)
            pic.save(f"{os.path.join(directory,str(dir))}\\{dir}_{n}.png")
    for dir in range(len(lis2)):
        if str(dir)+"_lnd" not in d:
            os.mkdir(os.path.join(directory,str(dir)+"_lnd"))
        for n,im in enumerate(lis2[dir]):
            with open(f"{directory}\{dir}_lnd\{n}_lnd.json", 'w') as f:
                json.dump({"keypoints": lis2[dir][n]}, f, indent=2)
            
    return




def getLandMarks(image):
    
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True)
    #mpDrawing = mp.solutions.drawing_utils

  
    if image.shape[2] != 3:
        image  = np.stack([image] * 3, axis=-1)
   

    results = pose.process(image)

    vertices = []
    edges = list(mp_pose.POSE_CONNECTIONS)
    keypoints = []
    if results.pose_landmarks:
        '''h, w, _ = image.shape
        for lm in results.pose_landmarks.landmark:
            x_px = lm.x * w
            y_px = lm.y * h
            vertices.append((x_px, y_px))'''
        for lm in results.pose_landmarks.landmark:
            keypoints.append({
                "x": lm.x,
                "y": lm.y,
                "z": lm.z,
                "visibility": lm.visibility
            })
        edges = [{"start": i, "end": j} for i, j in edges]
        keypoints.append(edges)
    return keypoints



'''def showImage(img, cmap=None, title=None):

    plt.figure(figsize=(6, 6))
    if img.ndim == 2:  # Grayscale image
        plt.imshow(img, cmap=cmap or 'gray')
    elif img.ndim == 3 and img.shape[2] in [3, 4]:  # RGB or RGBA
        plt.imshow(img)
    else:
        raise ValueError("Unsupported image shape: expected 2D or 3D with 3/4 channels.")
    
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()'''


#print(loadData("D:\dionigi\Documents\Python scripts\\aml2025Data\data"))
def main ():
    files=loadData("D:\dionigi\Documents\Python scripts\\aml2025Data\data")
    data=normDataset(files)
   # l=[]
    #for elem in data[1][10:20]:
   # v,e= getLandMarks(data[0][50])
      # l.append(v)
    #print(v)
    saveData(data[0],data[1],"D:\dionigi\Documents\Python scripts\\aml2025Data\dataNorm")

    #dims=countDims(files)
    #plotDims(dims)
    #print(max(dims))
    #print(min(dims))

   # rgb_img = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)
    #show_image(rgb_img, title="Random RGB Image")

    #f=applyCrop(rgb_img,0,200)

    #f1=applyCrop(f,1,200)

    #show_image(f1, title="Random RGB Image")
    #print(f1.shape)
    

    return

#main()