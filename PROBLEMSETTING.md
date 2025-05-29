### Yoga Pose Detection Model 

### Overview

The objective of this project is to develop a graph-based yoga pose detection pipeline utilizing a **Graph Neural Network (GNN)** architecture. To achieve this, we propose integrating **GNN** layers to fine-tune a pre-trained **Transformer** model, leveraging a <CLS> token for accurate classification of graph-represented poses. Our approach follows the **Mesh-Graphormer** architecture by incorporating **Graph Residual Blocks** into the attention mechanism of a pre-trained Transformer, such as **BERT**. The model will process two inputs: an image of a yoga pose and s stick-figure-like graph representation of the pose (encoded as positional embeddings), the output will be a classification of the detected yoga pose.  

---

### Formal problem formulation

Given an input space $\mathcal{X}$ consisting of pairs $(I, G)$, where:  
- $I$  denotes an image of a yoga pose,  
- $G = (V, E)$ represents a stick-figure-like graph structure of the pose, with nodes \( V \) (body joints) and edges $E$ (bone connections),  

the goal is to learn a mapping function $f: \mathcal{X} \rightarrow \mathcal{Y}$ that classifies the pose into one of $K$  predefined yoga categories $\mathcal{Y} = {y_1, y_2, \dots, y_K}$.

We propose a hybrid **Graph Neural Network (GNN)-Transformer** architecture to model $f$:  

1. **Input Encoding**:  
   - The image $I$ is processed via a vision backbone (e.g., CNN) to extract visual features.  
   - The graph $G$ is encoded as positional embeddings.

2. **Model Architecture**:  
   - Integrate **Graph Residual Blocks** into a pre-trained Transformer (**BERT**) by modifying its attention mechanism, following **Mesh-Graphormer**.  
   - The Transformer’s CLS token aggregates graph-structured features for classification.  

3. **Objective**:  
   Minimize the cross-entropy loss $\mathcal{L}$ over the training set $\mathcal{D}$:  
   $\mathcal{L} = -\sum_{(x,y) \in \mathcal{D}} \log p(y \mid x; \theta)$ <br>
   where $\theta$ denotes the model parameters.  

---

### Dataset

We plan to use two different publicly available datasets of yoga poses one containing 5 distinct poses and the other featuring 8 poses. These datasets will help us train and evaluate the performance of our **GNN-based** classification model on a diverse set of postures. However, since they do not come with joint-level annotations, we will label them ourselves. To achieve this, we will use **Google’s MediaPipe framework** to extract keypoints and generate stick-figure joint representations for each image. These will be used as the graph representation of our images.

**5-Pose Dataset**:
This dataset consists of images representing five different yoga poses. It can be accessed via Kaggle at the following link:
https://www.kaggle.com/datasets/niharika41298/yoga-poses-dataset/data

**8-Pose Dataset**:
This dataset includes images of eight different yoga poses and is available on GitHub:
https://github.com/Manoj-2702/Yoga_Poses-Dataset

---

### Model Inputs and Outputs
**Input**
- $I$: An RGB image of a person performing a yoga pose.
- $G = (V, E)$: A stick-figure-like pose graph where:
   - $V$ (nodes): Keypoints/joints extracted using MediaPipe.
   - $E$ (edges): Anatomical connections between body parts.

**Output**
A classification label $y \in \mathcal{Y} = \{y_1, y_2, \dots, y_K\}$ representing one of the $K$ yoga pose classes.

---

### Evaluation Metrics

- Since pose annotations (keypoints) are automatically generated using MediaPipe, we directly apply evaluation using $K$-fold cross-validation.
---

### Baselines

We will evaluate our approach against some baselines:

**Mesh Graphormer**:
[https://arxiv.org/abs/2104.03516](https://arxiv.org/pdf/2104.00272)

**Yoga-Pose-Classification-and-Skeletonization**
https://github.com/shub-garg/Yoga-Pose-Classification-and-Skeletonization



