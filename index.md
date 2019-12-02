## Brief Introduction towards Eyeblink Detection in the Wild

This project is an introduction of paper "Towards Real-time Eyeblink Detection in The Wild: Dataset, Theory and Practices", which is publiced in IEEE transactions on information forensics and security (tifs), In this paper, we shed the light to this research topic. A labelled eyeblink in the wild dataset (i.e., HUST-LEBW) of 673 eyeblink video samples (i.e., 381 positives, and 292 negatives) is first established, and formulate eyeblink detection task as a binary spatial-temporal pattern recognition problem. After locating and tracking human eye using SeetaFace engine and KCF {(Kernelized Correlation Filters)} tracker respectively, a modified LSTM model able to capture the multi-scale temporal information is proposed to execute eyeblink verification.

### Dataset
To reveal the "in the wild" characteristics, such as dramatic variations on human attribute, human pose, illumination, imaging viewpoint, and imaging distance, we propose an "in the wild" dataset named HUST-LEBW. The eyeblink samples in HUST-LEBW are collected from 20 different commercial movies, with different attribute informatiosn (i.e., name, filming location, style and premiere time). For instance, it consists of 673 eyeblink video clip samples (i.e., 381 positives, and 292 negatives) that captured from the unconstrained movies to reveal the characteristics of "in the wild". In addition, to reveal the real challenges in the wild, it also contians 91 untrimmed videos in testing set. This dataset can be download from [here](https://pan.baidu.com/s/1kyqVkzVVG-sYIaBOwr5pDQ).
![Image text](https://github.com/thorhu/Eyeblink-in-the-wild/blob/master/dataset/challenge.jpg)
```Dataset

# some snapshorts on HUST-LEBW

## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### environments & tools
1. opencv3.2.0
2. tensorflow 0.5.0
3. tensorflow-gpu 1.0.0
4. seetaface
5. visual studio 2015


Your Pages site will use ttttout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/thorhu/Eyeblink-in-the-wild/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
