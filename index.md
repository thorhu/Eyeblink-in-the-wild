
# Brief introduction towards eyeblink detection in the wild

.justify{
  width: 120px;
  text-align: justify;
}


This project is an introduction of paper "Towards Real-time Eyeblink Detection in The Wild: Dataset, Theory and Practices", which is publiced in IEEE transactions on information forensics and security (tifs), In this paper, we shed the light to this research topic. A labelled eyeblink in the wild dataset (i.e., HUST-LEBW) of 673 eyeblink video samples (i.e., 381 positives, and 292 negatives) is first established, and formulate eyeblink detection task as a binary spatial-temporal pattern recognition problem. After locating and tracking human eye using SeetaFace engine and KCF {(Kernelized Correlation Filters)} tracker respectively, a modified LSTM model able to capture the multi-scale temporal information is proposed to execute eyeblink verification.

![Image text](https://github.com/thorhu/Eyeblink-in-the-wild/blob/master/dataset/challenge.jpg)

### Dataset
To reveal the "in the wild" characteristics, such as dramatic variations on human attribute, human pose, illumination, imaging viewpoint, and imaging distance, we propose an "in the wild" dataset named HUST-LEBW. The eyeblink samples in HUST-LEBW are collected from 20 different commercial movies, with different attribute informatiosn (i.e., name, filming location, style and premiere time). For instance, it consists of 673 eyeblink video clip samples (i.e., 381 positives, and 292 negatives) that captured from the unconstrained movies to reveal the characteristics of "in the wild". In addition, to reveal the real challenges in the wild, it also contians 91 untrimmed videos in testing set. This dataset can be download from [here]().

### Environments
1. opencv3.2.0
2. tensorflow 0.5.0
3. tensorflow-gpu 1.0.0
4. seetaface
5. python 2.7.15

p.s. and some other neccessary datasets.

### Steps
Here, we will introduce steps to complete our code on details

#Initial steps to set up environments 

-To make things becomes easier, you may first install anaconda3, and the package can be find from [here](https://www.anaconda.com/download/), you can also download seetaface (c++ version)
 in [here](https://github.com/seetaface/SeetaFaceEngine.git). 
To set up opencv
```To set up opencv
conda install opencv
```

To set up tensorflow 
```To set up tensorflow 
gpu: pip install tensorflow-gpu==0.5.0
cpu: pip install tensorflow==0.5.0
```

#Step1: use seetaface with kcf to detect eye images, codes and guidence can be found from [here](https://github.com/thorhu/Eyeblink-in-the-wild/tree/master/detect_track_eye)

#Step2: after locate eye images, extract uniform-LBP feature from eye images, codes and guidence can be found from[here](https://github.com/thorhu/uniform_lbp-coding)

#Step3: dwnload codes from [here](https://github.com/thorhu/Eyeblink-in-the-wild/tree/master/codes/LSTM_combine_sphereface), and apply model:
```
cd codes/LSTM_combine_sphereface/

-train the model
python mylstm_10.py

-test the model
python mylstm_10_test.py
```
### Citation
if you use our dataset or code, please use the following citation:

[1] Guilei Hu, Yang Xiao, Zhiguo Cao, Lubin Meng, Zhiwen Fang, and Joey Tianyi Zhou. Towards real-time eyeblink detection in the wild: Dataset, theory and practices. arXiv preprint arXiv:1902.07891, 2019. 1, 2, 3, 4, 5, 6, 7, 8, 9

