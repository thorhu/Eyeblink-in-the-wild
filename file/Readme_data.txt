The newest code of Towards Real-time Eyeblink Detection in The Wild: Dataset, Theory and Practices and the HUST-LEBW dataset will always be available from https://thorhu.github.io/Eyeblink-in-the-wild/
--------------------------LICENSE------------------------
The dataset is provided for research purposes to a researcher. Please do not release the data or redistribute this link to anyone else without our permission. Contact {guilei_hu, Yang_Xiao}@hust.edu.cn if any question or any commercial use. Bibtex information for referencing is provided at the bottom of this document.

--------------------------DATASET SPLITS------------------------
To reveal the "in the wild" characteristics, such as dramatic variations on human attribute, human pose, illumination, imaging viewpoint, and imaging distance, we propose an "in the wild" dataset named HUST-LEBW. The eyeblink samples in HUST-LEBW are collected from 20 different commercial movies, with different attribute informatiosn (i.e., name, filming location, style and premiere time). For instance, it consists of 673 eyeblink video clip samples (i.e., 381 positives, and 292 negatives) that captured from the unconstrained movies to reveal the characteristics of "in the wild". Their resolutions are 1280 * 720 or 1456 * 600. Each video clips including a blink sample or unblink sample. Each sample have two time span obeying the chronological order. The first time span is 13 frames and the other is 10 frames. Every image’s eye location will be marked in a txt file for both time span. In some cases, these is only one eye visible, we will label this eye and the coordinate of the invisible eye will be (-1,-1). After we get coordinate of the all eyes, we use the coordinate to extract the visible eye images. In addition, to reveal the real challenges in the wild, it also contians 91 untrimmed videos in testing set.

The number of clips in the unbalanced and balanced datasets are shown below. 

Eyeblink video clip samples

-----------------------------------------------------------
| Eye	|  Action	                  |  Train set |  Test se   |
-----------------------------------------------------------
| Left	|  Eyeblink     	  |  243        |  122        |
| Left  	|  Non-eyeblink     	  |  181        |  98          |
| Right 	|  Eyeblink     	  |  256        |  126        |
| Right 	|  Non-eyeblink     	  |  190        |  98          |
-----------------------------------------------------------

Untrimmed videos

--------------------------------------------
| Num	|  Resolution              	  |
---------------------------------------------
| 91	| 1280 * 720 or 1456 * 600     	  | 
---------------------------------------------
--------------------------INTERMEDIATE RESULTS------------------------
Some intermediate results are provided for eyeblink video clip samples.

Txt_file
The landmarks' position of both eye are lables in land_10(13).txt for 10/13 pictures sample, the format is :
[Picture_id] [Left_x] [Left _y] [Right_x] [Right_y]

The pictures' id in samples are shown in sign_10(13).txt for 10/13 pictures sample, the format is :
[idx_1] ... [idx_n]

--------------------------FILMS RESOURCES------------------------

The films used to compile this dataset were:
1) Black Mirror
2) Game of Thrones
3) Blood Diamond
4) Chungking Express
5) Chivalrous Killer
6) A Clockwork Orange
7) Pirates of the Caribbean
8) Contratiempo
9) Ashes of Time
10) The Bourne Ultimatum
11) Mad Max 4
12) The Last Emperor
13) Farewell My Concubine
14) Léon
15) The Lord of the Rings: The Return of the King
16) Kill Bill Vol.1
17) Memories of Matsuko
18) A Chinese Fairy Tale
19) The Matrix
20) The Matrix Reloaded.

These films were split into train and test sets to emphasise generalisation across films. 

--------------------------CITATION------------------------

To simplify referencing the bibtex information for Hollywood 3D is below:

@article{hu2019towards,
  title={Towards Real-time Eyeblink Detection in The Wild: Dataset, Theory and Practices},
  author={Hu, Guilei and Xiao, Yang and Cao, Zhiguo and Meng, Lubin and Fang, Zhiwen and Zhou, Joey Tianyi},
  journal={arXiv preprint arXiv:1902.07891},
  year={2019}，
  keywords = {Eyeblink detection, eyeblink in the wild, spatial temporal pattern recognition, LSTM, appearance and motion},
  url = {https://arxiv.xilesou.top/pdf/1902.07891.pdf}
}

Copyright Guilei Hu, Yang Xiao are with National Key Laboratory of Science and Technology on Multi-Spectral Information Processing, School of Artificial Intelligence and Automation, Huazhong University of Science and Technology, China
