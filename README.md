This is the official code for CIKM 2023 Long Paper:
*HiPo: Detecting Fake News via Historical and Multi-Modal Analyses of Social Media Posts*
# Historical Posts-based Fake News Detector (HiPo)
We release the model HiPo, which is a context-driven fake news detection model using news environment. The following diagram illustrates HiPo's design. 
## Method
The overview of HiPo shown in the figure below. 
![overview](https://github.com/xy727682/CIKM2023HiPo/tree/main/pic/overview.png)
## Installation
The descriptions of enviroment requiments and configurations for the HiPo will be added soon. Before that, please refer to the experiment chapter in 4.1 for the installation of HiPo.  
## Experimental Results
HiPo's Performance is shown in the graph below. 
![results](https://github.com/xy727682/CIKM2023HiPo/tree/main/pic/experimental.png "results")
# Datasets
Since the dataset of this work comes from many previous works and is not easy to be disclosed directly, two ways of obtaining the data are provided:
## Self Reconstruction
Build the dataset locally by referring to the following instructions (recommended: you can repartition the data on your own, which is more flexible and facilitates the data analysis):
- The Fakeddit dataset comes from [r/Fakeddit](https://github.com/entitize/Fakeddit "r/Fakeddit"), of which a multi-model part is used in this work, where according to the UTC, <code>1570000088</code> to <code>1570999956</code> is the training set, larger than <code>1571349963</code> is the validation set, <code>1571000033</code> to <code>1571349963</code> is the test set, and <code>1212297305</code> to <code>1569999810</code> is the environment set. 
- The IFND dataset is from [IFND](https://link.springer.com/article/10.1007/s40747-021-00552-1?utm_source=xmol&utm_medium=affiliate&utm_content=meta&utm_campaign=DDCN_1_GL01_metadata "IFND"), which is repartitioned in this work, where according to date, 2020.4 to 2020.11 is the training set, 2020.12 to 2021.1 is the validation set, 2020.11 to 2020.12 is the test set, and 2004.8 to 2020.4 is the environment set
- The Twitter dataset comes from [Twitter15/16](https://www.dropbox.com/s/7ewzdrbelpmrnxu/rumdetect2017.zip?dl=0 "Twitter15/16") and [MuMiN](https://mumin-dataset.github.io/ "MuMiN") and [NEP](https://github.com/ICTMCG/News-Environment-Perception/ "NEP"), 2020.7.5 to 2020.9.8 is the training set, 2020.9.25 to 2021.5.27 is the validation set, 2020.9.9 to 2020.9.24 is the test set, and 2016.6.20 to 2020.9.7 is the environment set
- The Weibo dataset comes from [Weibo](http://alt.qcri.org/~wgao/data/rumdect.zip) and [NEP](https://github.com/ICTMCG/News-Environment-Perception/ "NEP"), where as dated, 2012.8.27 to 2015.4.21 is the training set, 2015.10.28 to 2016.1.24 is the validation set, 2015.4.21 to 2015.10.28 is the test set, and 2012.8.27 to 2016.1.13 is the environment set. 
We summarize the begin points and end points of different partitioned sets in the following table:

| **Dataset**  |       **Train**       |      **Validate**     |        **Test**       |    **Environment**    |
|--------------|:---------------------:|:---------------------:|:---------------------:|:---------------------:|
| **Fakeddit** | 2019/10/02-2019/10/13 | 2019/10/17-           | 2019/10/13-2019/10/17 | 2008/06/01-2019/10/02 |
| **IFND**     | 2020/04-2020/11       | 2020/12-2021/01       | 2020/11-2020/12       | 2004/08-2020/04       |
| **Twitter**  | 2020/07/05-2020/09/08 | 2020/09/25-2021/05/27 | 2020/09/09-2020.09.24 | 2016/06/20-2020/09/07 |
| **Weibo**    | 2012/08/27-2015/04/21 | 2015/10/28-2016/01/24 | 2015/04/21-2015/10/28 | 2012/08/27-2016/01/13 |
Table 1. The start and end dates of the publication of the news contained in each of the divided sets. 
###### NOTE: 
1. the selection of historical news should be performed using the time order, and avoid using the future news for model training. 
2. There are multiple instances of dataset items where images unaccessible to the public and scientific use, for which we removed the news items with their image information out of our reach during the experiment preparations, leading to reduced sizes of collected datasets compared with original released ones. For codes related to this issue, please contact the author (see the next section for contact information). 
## Direct Adaptation
Directly use the data processed by pre-trained model (not recommended: no raw data, infeasible for data analysis, only suitable for quick verification of the model performance)
## Acquiring Datasets Employed in the HiPo Paper
Due to the large amount of data, if you need to use the pre-trained data, please contact 201922180266@mail.sdu.edu.cn or 2023244076@tju.edu.cn by email, please note the dataset you need, the way you want to use it and the way you want to get the data (Baidu or Google Drive), or other specific issues mentioned above. We will check the emails from time to time, so please be patient.
# Citation
If you find this code useful in your research, you can cite our paper:
```markdown
@article{Xiao2023HiPoDF,
  title={HiPo: Detecting Fake News via Historical and Multi-Modal Analyses of Social Media Posts},
  author={Tianshu Xiao and Sichang Guo and Jingcheng Huang and Riccardo Spolaor and Xiuzhen Cheng},
  journal={Proceedings of the 32nd ACM International Conference on Information and Knowledge Management},
  year={2023},
}
```
