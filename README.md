# 21-SLT-project

## MLVU Final project : Sign Language Translation(SLT) Transformer

### Project Introduction
This repository contains implementation of my research based on the work during the 2021 Machine Larning for Visual Unerstanding course. You can see the final [presentation](https://www.youtube.com/watch?v=WEfdDxsFOCc&list=PL0E_1UqNACXDaCMnwgiM75SKIpHf2mpif&index=7) and [report](http://vip.joonseok.net/courses/mlvu_2021_1/projects/team07.pdf) which introduces our base model(seq2seq+attention). <br>


![Block diagram of our model](https://github.com/Seunghoon-Yi/21-SLT-project/assets/57204784/d7b15ece-cfe5-4106-b114-2ced3009c37c)


<br>

To improve speed and achieve keypoit-free SLT, We constructed our model with three components. <br>
* **Lightweight Video encoder** <br>
3d CNN(C3D, R(2+1)D, S3D)s to encode the video, returns the encoded features. <br>
* **Transformer Encoder** <br>
Contextualizes the input features and returns **glosses**, which are word-level elements of the sign language. <br>
* **Sign Language Decoder88 <br>
The transformer decoder learn the relations between glosses and spoken language, and returns the translated text. <br>
The model is trained with [RWTH-PHOENIX-Weather 2014 dataset](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/) which contains sign language videos from 5~30 seconds, and recorded a **BLEU-4 score of 14.45**.

<br>

### Requirements
* cuda >= 11.3
* python >= 3.7
* pytorch >= 1.8.1
  * pytorch-model-summary
* torchvision >= 0.9.1
* tqdm
  
<br>

to run training, run as <code>python train.py</code> after downloading the dataset to your working directory. 
