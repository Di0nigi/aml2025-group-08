### Yoga Pose Detection Model 

### Overview

The objective of this project is to develop a graph-based yoga pose detection pipeline utilizing a **Graph Neural Network (GNN)** architecture. To achieve this, we propose integrating **GNN** layers to fine-tune a pre-trained **Transformer** model, leveraging a **$<CLS>$** token for accurate classification of graph-represented poses. Our approach follows the **Mesh-Graphormer** architecture by incorporating **Graph Residual Blocks** into the attention mechanism of a pre-trained Transformer, such as **BERT**. The model will process two inputs: an image of a yoga pose and s stick-figure-like graph representation of the pose (encoded as positional embeddings), the output will be a classification of the detected yoga pose.  

---

### Formal problem formulation

Given an input space \( \mathcal{X} \) consisting of pairs \((I, G)\), where:  
- \( I \) denotes an image of a yoga pose,  
- \( G = (V, E) \) represents a stick-figure-like graph structure of the pose, with nodes \( V \) (body joints) and edges \( E \) (bone connections),  

the goal is to learn a mapping function \( f: \mathcal{X} \rightarrow \mathcal{Y} \) that classifies the pose into one of \( K \) predefined yoga categories \( \mathcal{Y} = \{y_1, y_2, \dots, y_K\} \).  

We propose a hybrid **Graph Neural Network (GNN)-Transformer** architecture to model \( f \):  

1. **Input Encoding**:  
   - The image \( I \) is processed via a vision backbone (e.g., CNN) to extract visual features.  
   - The graph \( G \) is encoded as positional embeddings.

2. **Model Architecture**:  
   - Integrate **Graph Residual Blocks** into a pre-trained Transformer (**BERT**) by modifying its attention mechanism, following **Mesh-Graphormer**.  
   - The Transformer’s **$<CLS>$** token aggregates graph-structured features for classification.  

3. **Objective**:  
   Minimize the cross-entropy loss \( \mathcal{L} \) over the training set \( \mathcal{D} \):  
   \[
   \mathcal{L} = -\sum_{(x,y) \in \mathcal{D}} \log p(y \mid x; \theta),
   \]  
   where \( \theta \) denotes the model parameters.  

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
The first model will receive an image of a person performing a yoga pose and will generate multiple heatmaps, one for each joint in the body. Each heatmap highlights the regions where the corresponding joint is most likely to be located. Using these heatmaps, we will construct a stick-figure representation of the pose by connecting the predicted joint positions.

The second model will then take this stick-figure representation as input and classify the pose by predicting the top n most probable pose classes. This two-stage approach allows us to decouple pose estimation from pose classification, potentially improving overall accuracy.

---

### Evaluation Metrics
For the evaluation of our models, we will primarily use validation set accuracy, as our approach is based on supervised learning. This metric will help us assess how well the classification model performs on unseen data. In addition, for the pose fitting task, we will use Object Keypoint Similarity (OKS) to quantitatively evaluate the accuracy of predicted joint locations, which is defined as follows:

$\text{OKS} = \frac{\sum_i \exp\left(-\frac{d_i^2}{2s^2k_i^2}\right) \cdot \delta(v_i > 0)}{\sum_i \delta(v_i > 0)}$


where $d_i$  is the Euclidean distance between the predicted and true keypoints, $s$ is the object scale, $k_i$ is a per-keypoint constant that controls falloff, and $v_i$ indicates the visibility of each keypoint. To evaluate the generalization ability of our pose fitting model, we will include yoga poses in the test set that were not seen during training. This will allow us to assess how well the model generalizes to new poses.

---

### Baselines

We will evaluate our approach against two baseline models. The first baseline is the **TokenPose** architecture, which serves as the main inspiration for our project. The second is the baseline method used for comparison in the TokenPose paper itself.

**TokenPose Learning Keypoint Tokens for Human Pose Estimation**:
https://arxiv.org/abs/2104.03516

**Simple Baselines for Human Pose Estimation and Tracking**:
https://arxiv.org/abs/1804.06208

These comparisons will help contextualize the performance of our models and demonstrate any improvements achieved through our proposed modifications.

