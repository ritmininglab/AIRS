# Adaptive Important Region Selection with Reinforced Hierarchical Searching for Dense Object Detection


## Introduction

Existing state-of-the-art dense object detection techniques (e.g., GFOCAL and ATSS) tend to produce false positive
bounding boxes during the inference process as they are designed
to ensure a high recall. Consequently, these models could lead
to a low precision by including many noisy and unnecessary
patches resulting from attending to the low level details in a
given image. To address this limitation, we propose to conduct
novel evidential deep Q-learning that leverages a principled
hierarchical search strategy to explore image patches via a topdown fashion. The reinforcement learning (RL) agent is guided
by a uniquely designed evidential Q-value that evaluates both the
patch quality and evidential uncertainty as a way to optimally
balance exploitation and exploration of a potential large search
space with image patches containing different levels of details.
In particular, patch quality is quantifed through a customized
reward function defned based on the ground truth bounding
box information to avoid searching on the patches with a fner
granularity, where the objects are less likely to be present. This
helps to greatly reduce false positive bounding boxes coming
from the fne-level granularity. Meanwhile, evidential uncertainty
will encourage the RL agent to continue searching any patch
that may potentially contain any object so that important ones
won’t be missed to sacrifce the recall. The effective hierarchical
searching resulting from proper exploration and exploitation
strategy helps to search all potential patches while avoiding
the low-level unnecessary patches. Experiments conducted on
multiple dense object detection datasets demonstrate that our
approach improves the precision without sacrifcing the recall.

Here we show the training and test diagram of our RL agent to filter out small noisy local boxes.

<img src="Training of RL agent for region filtering.jpg" width="500" height="300" align="middle"/>
<img src="Test for RL agent in dense object detection.jpg" width="500" height="300" align="middle"/>



## Installation


Please refer to [INSTALL.md](docs/INSTALL.md) for installation and dataset preparation.



## Get Started


Please see [GETTING_STARTED.md](docs/GETTING_STARTED.md) for the basic usage of MMDetection.



## Train


```python

# assume that you are under the root directory of this project,

# and you have activated your virtual environment if needed.

# and with COCO dataset in 'data/coco/'


python tools/train.py configs/air_r50_1x.py 3 --validate

```


## Inference


```python

python tools/test.py configs/air_r50_1x.py work_dirs/air_r50_1x/epoch_24.pth 3 --eval bbox

```


## Speed Test (FPS)


```python

CUDA_VISIBLE_DEVICES=0 python3 ./tools/benchmark.py configs/air_r50_1x.py work_dirs/air_r50_1x/epoch_24.pth

```


## Dataset
[MS COCO](https://cocodataset.org/#home)
[Pascal VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)


## Pretrained-RL agent Model

[Map-Crop+EQL](https://drive.google.com/file/d/1K_WQe2xrTQWZWsnrfq3BmwIGcW5H_gor/view?usp=sharing)
[Vanilla+EQL](https://drive.google.com/file/d/1K_WQe2xrTQWZWsnrfq3BmwIGcW5H_gor/view?usp=sharing)
