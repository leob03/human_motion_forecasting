<p align="center">
<img src=./img/BRG.png width=40% height=40%>
</p>

# Human Motion Forecasting for Dynamic Human Robot Collaboration

This repository contains a Neural Network model for Human Motion Forecasting in the context of Dynamic Human-Robot Collaboration as well as visualisation tools to compare state of the art models on real-time skeleton data captured with an Azure Kinect leveraging ROS 3D visualizer software RViz. The network architecture relies on Graph Convolutional Network, Self-Attention Layers and LSTMs. This code is the implementation of a research project conducted at the [Barton Research Group](https://brg.engin.umich.edu/research/human-digital-twins/).

&nbsp;

<p align="center">
  <img src=./img/combined.png width="870" height="280">
</p>

# Content

[***Objective***](https://github.com/leob03/human_motion_forecasting#objective)

[***Comparisons***](https://github.com/leob03/human_motion_forecasting#comparisons)

[***Concepts***](https://github.com/leob03/human_motion_forecasting#concepts)

[***Architecture***](https://github.com/leob03/human_motion_forecasting#architecture)

[***Dependencies***](https://github.com/leob03/human_motion_forecasting#dependencies)

[***Getting started***](https://github.com/leob03/human_motion_forecasting#getting-started)

[***References***](https://github.com/leob03/human_motion_forecasting#references)

[***Acknowledgments***](https://github.com/leob03/human_motion_forecasting#acknowledgments)

[***License***](https://github.com/leob03/human_motion_forecasting#license)

# Objective

**With a simple RGB-D Camera predict the future motion of a human in a Smart Manufacturing workspace of to ensure safety in human-robot interactions.**

# Comparisons

During this project we studied and implemented multiple methods for Human Motion Forecasting, compared their performance (accuracy, speed and memory consumption) on Human 3.6M but most of all on our own Human-Robot Collaboration workspace set up. Indeed, we wanted to have an idea of how these models performed on real-time data. We compared the results of these implenetations on real-time skeleton data captured with the Azure Kinect SDK and worked on a visualization tool to compare those results. Finally, based on this previous study chose one method as a benchmark ("History Repeats Itself" by Wei Mao) and tried multiple different types of architecture improvements to obtain even better results.

These are some visuals of the comparative results obtained :

&nbsp;

<p align="center">
  <img src="./gif/STS-GCN.gif" alt="Image Description" width="250" height="260">
  <br>
  Comparisons with [STS-GCN](https://github.com/FraLuca/STSGCN) (Ground-Truth in Blue, STS-GCN in Purple, and our model in Green):
</p>

&nbsp;

<p align="center">
<!--   Some visual results of : -->
  <img src="./gif/siMLPe.gif" alt="Image Description" width="250" height="280">
  <br>
  Comparisons with [siMLPe](https://github.com/dulucas/simlpe) (Ground-Truth in Blue, siMLPe in Pale Blue, and our model in Green):
</p>

&nbsp;

<p align="center">
<!--   Some visual results of : -->
  <img src="./gif/HRI.gif" alt="Image Description" width="250" height="270">
    <br>
    Comparisons with [HRI](https://github.com/wei-mao-2019/HisRepItself) (Ground-Truth in Blue, HRI in Red, and our model in Green):
</p>

&nbsp;

Since it is impossible to have a groundtruth of future movements,we "predicted the present", i.e. we only give the model informations about the past (skeleton data recorded up until 400ms away) and tried to predict the current motion. The delay for each predictive model compared to the ground-truth is due to the inference time, which is slower than the refreshment rate of the visual data published at 15 FPS.

# Concepts

Quick summary of the main concepts exploited in this project:

## Graph Convolutional Networks (GCN)

- **Description**: Neural networks designed for graph-structured data. Well-fitted for skeleton data where the graph could be either one body pose (to learn on the joint connectivity at each pose) or a set of body poses (to learn temporal evolution and speed of each joints).
- **Technical Aspects**: GCNs generalize the convolution operation from regular grids to graphs. They work by aggregating feature information from neighbors in a graph, $H^{(l+1)} = \sigma( D^{-1/2} \hat{A} D^{-1/2} H^{(l)} W^{(l)} )$, where $H^{(l)}$ is the feature representation at layer $l$, $\hat{A}$ is the adjacency matrix with added self-connections, $D$ is the degree matrix, $W^{(l)}$ is the weight matrix for layer $l$, and $\sigma$ is the non-linear activation function.

## Motion Attention

- **Description**: A specialized form of self-attention, inspired by transformers, tailored for identifying similar motion sub-sequences in historical data to predict future movements. Unlike traditional attention mechanisms in transformers that focus on enhancing sequence modeling, motion attention specifically targets the recognition of repetitive human motion patterns over time.
- **Technical Aspects**: The model maps a query (the last observed sub-sequence) and key-value pairs (historical and future motion representations) to an output. It divides the motion history into sub-sequences, using the first part of each as a key and the entire sub-sequence as a value. The attention scores are computed using the formula  $a_i = q k_i^T$, where $q$ is the query, and $k_i$ are the keys. These scores are normalized by their sum, avoiding the gradient vanishing issue often encountered with softmax in traditional attention mechanisms. The final output $U$ is calculated as a weighted sum of the values $U = \sum_{i=1}^{N-M-T+1} a_i V_i$, where $V_i$ are the values transformed into trajectory space using DCT. This process captures partial motion similarities, enabling the model to predict future poses effectively.


## Temporal Encoding and DCT (Discrete Cosine Transform)

- **Description**: Techniques for encoding temporal information in neural networks.
- **Technical Aspects**: DCT transforms a sequence of values into components of different frequencies, $X_k = \sum_{n=0}^{N-1} x_n \cos\left[\frac{\pi}{N} (n + \frac{1}{2})k\right]$, where $X_k$ is the kth coefficient and $x_n$ is the nth element of the input sequence.

## LSTMs (Long Short-Term Memory networks)

- **Description**: A type of recurrent neural network specialized in remembering long-term dependencies.
- **Technical Aspects**: Incorporates gates that regulate the flow of information: an input gate, an output gate, and a forget gate. These gates determine what information should be retained or discarded at each step in the sequence.

## 2 Channel Spatio-Temporal Network

- **Description**: A network architecture for simultaneous spatial and temporal data processing.
- **Technical Aspects**: This architecture processes spatial and temporal components separately and then fuses them. The spatial channel captures static features, while the temporal channel captures dynamic changes, often using a sequence of frame differences or optical flow techniques.

# Architecture

Using most of the concepts precedently defined, here is an overview of the Neural Network architecture of our Method:
<p align="center">
  <img src=./img/arch.png width="600" height="400">
</p>

# Dependencies
- PyTorch >= 1.5 (ours is 1.9.0)
- Numpy
- CUDA >= 10.1
- Easydict
- pickle
- einops
- scipy
- six
- ROS Noetic


# Getting started

## Test on common datasets

Download all the data and put them in the `./data` directory.

[H3.6M](https://drive.google.com/file/d/15OAOUrva1S-C_BV8UgPORcwmWG2ul4Rk/view?usp=share_link)

[Original stanford link](http://www.cs.stanford.edu/people/ashesh/h3.6m.zip) has crashed, this link is a backup.

Directory structure:
```shell script
data
|-- h36m
|   |-- S1
|   |-- S5
|   |-- S6
|   |-- ...
|   |-- S11
```

[AMASS](https://amass.is.tue.mpg.de/)

Directory structure:
```shell script
data
|-- amass
|   |-- ACCAD
|   |-- BioMotionLab_NTroje
|   |-- CMU
|   |-- ...
|   |-- Transitions_mocap
```

[3DPW](https://virtualhumans.mpi-inf.mpg.de/3DPW/)

Directory structure: 
```shell script
data
|-- 3dpw
|   |-- sequenceFiles
|   |   |-- test
|   |   |-- train
|   |   |-- validation
```


## Test with your own RGB-D Camera
(coming soon)

# References

- Martinez, J., Black, M. J., & Romero, J. (2017). "On human motion prediction using recurrent neural networks."
- Li, C., Zhang, Z., Lee, W. S., & Lee, G. H. (2018). "Convolutional Sequence to Sequence Model for Human Dynamics."
- Ruiz, A. H., Gall, J., & Moreno-Noguer, F. (2019). "Human Motion Prediction via Spatio-Temporal Inpainting."
- Mao, W., Liu, M., Salzmann, M., & Li, H. (2020). "Learning Trajectory Dependencies for Human Motion Prediction."
- Mao, W., Liu, M., & Salzmann, M. (2020). "History Repeats Itself: Human Motion Prediction via Motion Attention."
- Sofianos, T., Sampieri, A., Franco, L., & Galasso, F. (2021). "Space-Time-Separable Graph Convolutional Network for Pose Forecasting."
- Aksan, E., Kaufmann, M., Cao, P., & Hilliges, O. (2021). "A Spatio-temporal Transformer for 3D Human Motion Prediction."
- Guo, W., Du, Y., Shen, X., Lepetit, V., Alameda-Pineda, X., & Moreno-Noguer, F. (2022). "Back to MLP: A Simple Baseline for Human Motion Prediction."
- Dang, L., Nie, Y., Long, C., Zhang, Q., & Li, G. (2022). "MSR-GCN: Multi-Scale Residual Graph Convolution Networks for Human Motion Prediction."
- Ma, T., Nie, Y., Long, C., Zhang, Q., & Li, G. (2022). "Progressively Generating Better Initial Guesses Towards Next Stages for High-Quality Human Motion Prediction."
- Li, M., Chen, S., Zhang, Z., Xie, L., Tian, Q., & Zhang, Y. (2022). "Skeleton-Parted Graph Scattering Networks for 3D Human Motion Prediction."
- Nargund, A. A., & Sra, M. (2023). "SPOTR: Spatio-temporal Pose Transformers for Human Motion Prediction."
- Gao, X., Du, S., Wu, Y., & Yang, Y. (2023). "Decompose More and Aggregate Better: Two Closer Looks at Frequency Representation Learning for Human Motion Prediction."

# Acknowledgments

The predictor model code is originally adapted from [HRI](https://github.com/wei-mao-2019/HisRepItself).

Some of our evaluation code and data process code (on Human3.6) was adapted from [Residual Sup. RNN](https://github.com/una-dinosauria/human-motion-prediction) by [Julieta](https://github.com/una-dinosauria). 

We also use those github repositories to make our own implementation of some SOTA methods:
- [siMLPe (WACV 2023)](https://github.com/dulucas/simlpe)
- [STS-GCN (ICCV 2021)](https://github.com/FraLuca/STSGCN)
- [LTD (ICCV 2019)](https://github.com/wei-mao-2019/LearnTrajDep)
- [HRI (ECCV 2020)](https://github.com/wei-mao-2019/HisRepItself)
- [Residual Sup. RNN (CVPR 2017)](https://github.com/una-dinosauria/human-motion-prediction)


# License
This code is distributed under an [MIT LICENSE](LICENSE).

Note that our code depends on other libraries, including CLIP, SMPL-X, PyTorch3D, and uses datasets that each have their own respective licenses that must also be followed.
