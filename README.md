# 21-SLT-project

## MLVU Final project : Sign Language Translation(SLT) Transformer

### Project Introduction
This repository contains implementation of my research based on the work during the 2021 Machine Larning for Visual Unerstanding course. You can see the final [presentation](https://www.youtube.com/watch?v=WEfdDxsFOCc&list=PL0E_1UqNACXDaCMnwgiM75SKIpHf2mpif&index=7) and [report](http://vip.joonseok.net/courses/mlvu_2021_1/projects/team07.pdf) which introduces our base model(seq2seq+attention). <br>

![Block diagram of our model](https://github.com/Seunghoon-Yi/21-SLT-project/assets/57204784/4baab743-d4ef-4257-9c4c-960862775eff)

<br>

To improve speed and achieve keypoit-free SLT, The model is constructed with a lightweight 3d CNN(C3D, R(2+1)D, S3D) to encode the video, and the transformer encoder returns **glosses**, which are word-level elements of the sign language. Finally, the decoder is trained to learn the relations between glosses and spoken language, and returns the translated text. The model is trained with [RWTH-PHOENIX-Weather 2014 dataset](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/) which contains sign language videos from 5~30 seconds, and recorded a **BLEU-4 score of 14.45**.

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
