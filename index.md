# Brief introduction towards eyeblink detection in the wild

Effective and real-time eyeblink detection is of wide-range applications, such as deception detection, drive fatigue
detection, face anti-spoofing, etc. Although numerous of efforts have already been paid, most of them focus on addressing the eyeblink detection problem under the constrained indoor conditions with the relative consistent subject and environment setup. Nevertheless, towards the practical applications eye blink detection in the wild is more required, and of greater challenges. In order to formulate these challenges, we propose a work named "Towards Real-time Eyeblink Detection in The Wild: Dataset, Theory and Practices", which is publiced in IEEE transactions on information forensics and security (**TIFS**). In this paper,  a labelled eyeblink in the wild dataset (i.e., HUST-LEBW) of 673 eyeblink video samples (i.e., 381 positives, and 292 negatives) is first established, and formulate eyeblink detection task as a binary spatial-temporal pattern recognition problem. After locating and tracking human eye using SeetaFace engine and KCF {(Kernelized Correlation Filters)} tracker respectively, a modified LSTM model able to capture the multi-scale temporal information is proposed to execute eyeblink verification.

# Dataset
To reveal the "in the wild" characteristics, such as dramatic variations on human attribute, human pose, illumination, imaging viewpoint, and imaging distance, we propose an "in the wild" dataset named HUST-LEBW. The eyeblink samples in HUST-LEBW are collected from 20 different commercial movies, with different attribute informatiosn (i.e., name, filming location, style and premiere time). For instance, it consists of 673 eyeblink video clip samples (i.e., 381 positives, and 292 negatives) that captured from the unconstrained movies to reveal the characteristics of "in the wild". Their resolutions are 1280 * 720 or 1456 * 600. Each video clips including a blink sample or unblink sample. Each sample have two time span obeying the chronological order. The first time span is 13 frames and the other is 10 frames. Every image’s eye location will be marked in a txt file for both time span. In some cases, these is only one eye visible, we will label this eye and the coordinate of the invisible eye will be (-1,-1). After we get coordinate of the all eyes, we use the coordinate to extract the visible eye images. In addition, to reveal the real challenges in the wild, it also contians 91 untrimmed videos in testing set. This dataset can be download from [BaiduYun](https://pan.baidu.com/s/1_xJPfKEJYI3S9adOlagHTg) or [Mega](https://mega.nz/#F!JQRUTSgS!1uM9jh8Oulw-BGrZUfaCQQ).

![Octocat](https://raw.githubusercontent.com/thorhu/Eyeblink-in-the-wild/master/dataset/challenge.jpg)


### Copyright
The dataset is provided for research purposes to a researcher only and not for any commercial use. Please do not release the data or redistribute this link to anyone else without our permission. Contact {guilei_hu, Yang_Xiao}@hust.edu.cn if any question. [License](https://github.com/thorhu/Eyeblink-in-the-wild/blob/master/file/License.txt) and [readme](https://github.com/thorhu/Eyeblink-in-the-wild/blob/master/file/Readme_data.txt) file can also be found from its link.

If you use this dataset, please cite it as following, also the bibtex can be found from [here](https://arxiv.org/abs/1902.07891)
[1] Guilei Hu, Yang Xiao, Zhiguo Cao, Lubin Meng, Zhiwen Fang, and Joey Tianyi Zhou. Towards real-time eyeblink detection in the wild: Dataset, theory and practices. arXiv preprint arXiv:1902.07891, 2019. 1, 2, 3, 4, 5, 6, 7, 8, 9



### Acknowledgment (File resource)
Samples are extracted from the following films:
Black Mirror；Game of Thrones；Blood Diamond；Chungking Express；Chivalrous Killer；A Clockwork Orange；Pirates of the Caribbean；Contratiempo；Ashes of Time；The Bourne Ultimatum；Mad Max 4；The Last Emperor；Farewell My Concubine；Léon；The Lord of the Rings: The Return of the King；Kill Bill Vol.1；Memories of Matsuko；A Chinese Fairy Tale;The Matrix; The Matrix Reloaded.

# Method and codes
## Environments
1. opencv3.2.0
2. tensorflow 0.5.0
3. tensorflow-gpu 1.0.0
4. seetaface
5. python 2.7.15

P.s. and some other neccessary datasets.

## Steps
Here, we will introduce steps to complete our code on details

### Initial steps to set up environments 

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

### Step1: eye images extraction

Use seetaface with kcf to detect eye images, codes and guidence can be found from [here](https://github.com/thorhu/Eyeblink-in-the-wild/tree/master/detect_track_eye)

### Step2: feature extraction

After locating eye images, extract uniform-LBP feature by eye images, codes and guidence can be found from[here](https://github.com/thorhu/uniform_lbp-coding)

### Step3: eyeblink verification

After extracting uniform-LBP feature, we use MS-LSTM to train and verify eyeblink detection, codes can be downloaded from [here](https://github.com/thorhu/Eyeblink-in-the-wild/tree/master/codes/LSTM_combine_sphereface), and follow these steps to apply our model:
```
cd codes/LSTM_combine_sphereface/

-train the model
python mylstm_10.py

-test the model
python mylstm_10_test.py
```

# Time computation

![Branching](https://github.com/thorhu/Eyeblink-in-the-wild/blob/master/dataset/time.png?raw=true)

# Successful testing samples

[video](http://v.youku.com/v_show/id_XNDQ2NDU0NTAwOA==.html?spm=a2h3j.8428770.3416059.1)
