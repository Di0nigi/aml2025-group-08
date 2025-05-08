### Yoga Pose Detection Pipeline 

### Overview

The aim of this project is to build an image-based yoga-pose detection pipeline using two models: a **Visual Transformer-based** model for pose fitting on our images and a **GNN-based** model to classify these images into their respective poses. To solve this problem, we plan to fine-tune a pretrained transformer model and adapt it to a joint-recognition task by following the *TokenPose: Learning Keypoint Tokens for Human Pose Estimation* architecture and tailoring it to our use case.
For pose classification, we will train a **GNN** from scratch on a finite set of poses to harness the inherent structural properties of pose-fitted data.

---

### Formal problem formulation

**input data**

Given a dataset **$D=\{(X_i, Y_i)\}^k_{i=1} $**, where:

- **Input Image**:  
  $X_i \in {R}^{H \times W \times 3}$ is an RGB image of a yoga pose.

- **Keypoint Annotations**:  
  $Y_i \in {R}^{N \times 2}$ contains 2D coordinates for $N$ body joints.


**Objective**

Build a two-stage pipeline consisting of:

Pose Estimation

- Use a Visual Transformer-based model to detect keypoints from $X_i$.
- **Output**: Heatmaps $H_i \in {R}^{\hat{H} \times \hat{W} \times N}$ for each keypoint.

Pose Classification

- Use a Graph Neural Network (GNN) to classify the detected pose into one of $C$ predefined yoga poses.

- **Input**:
  - A graph $ G_i = (V_i, E_i) $, where:
    - $ V_i \in \mathbb{R}^{N \times d}$: Each node represents a detected joint (keypoint), encoded by its 2D coordinates and optionally other features (e.g., confidence scores).
    - $ E_i $: Set of edges capturing anatomical or learned relationships between joints.
- **Output**:
  - A class probability vector $ \hat{p}_i \in \mathbb{R}^{C} $ over $ C $ predefined yoga poses.
  - Final predicted class:  
    $$
    \hat{c}_i = \arg\max_j \hat{p}_i^{(j)}
    $$

---

### Dataset

For the classification task, we plan to use two different publicly available datasets of yoga poses one containing 5 distinct poses and the other featuring 8 poses. These datasets will help us train and evaluate the performance of our **GNN-based** classification model on a diverse set of postures.

**5-Pose Dataset**:
This dataset consists of images representing five different yoga poses. It can be accessed via Kaggle at the following link:
https://www.kaggle.com/datasets/niharika41298/yoga-poses-dataset/data

**8-Pose Dataset**:
This dataset includes images of eight different yoga poses and is available on GitHub:
https://github.com/Manoj-2702/Yoga_Poses-Dataset

For the pose fitting task, we plan to use the same two yoga pose datasets mentioned earlier. However, since they do not come with joint-level annotations, we will label them ourselves. To achieve this, we will use **Google’s MediaPipe framework** to extract keypoints and generate stick-figure joint representations for each image. These labeled outputs will serve as the ground truth for training our pose fitting model.

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

