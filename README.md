# Unsupervised Detection of Distinctive Regions on 3D Shapes
by [Xianzhi Li](https://nini-lxz.github.io/), [Lequan Yu](https://yulequan.github.io/), [Chi-Wing Fu](https://www.cse.cuhk.edu.hk/~cwfu/), [Daniel Cohen-Or](https://www.cs.tau.ac.il/~dcor/) and [Pheng-Ann Heng](http://www.cse.cuhk.edu.hk/~pheng/).


![Distinctive regions detected by our method unsupervisedly on various 3D models](https://github.com/nini-lxz/Unsupervised-Shape-Distinction-Detection/blob/master/images/vis_examples.png)


### Introduction
This repository is for our ACM Transactions on Graphics (TOG) 2020 paper '[Unsupervised Detection of Distinctive Regions on 3D Shapes](https://arxiv.org/abs/1905.01684)'. In this paper, we present a novel approach to learn and detect distinctive regions on 3D shapes. Unlike previous works, which require labeled data, our method is unsupervised. We conduct the analysis on point sets sampled from 3D shapes, then formulate and train a deep neural network for an unsupervised shape clustering task to learn local and global features for distinguishing shapes with respect to a given shape set. To drive the network to learn in an unsupervised manner, we design a clustering-based nonparametric softmax classifier with an iterative re-clustering of shapes, and an adapted contrastive loss for enhancing the feature embedding quality and stabilizing the learning process. By then, we encourage the network to learn the point distinctiveness on the input shapes. We extensively evaluate various aspects of our approach and present its applications for distinctiveness-guided shape retrieval, sampling, and view selection in 3D scenes.

### Usage
(1) You can directly test your sampled point clouds using our previously-trained network:  
The testing code (test.py) is provided inside the `code` folder, and our trained network based on the ModelNet40 inter-class dataset is provided inside the `trained_model` folder.
After running, the network will output per-point distinctiveness value for each input point.

(2) You can also re-train our network either using your own training samples or our provided [training dataset](https://drive.google.com/open?id=1bG2UpN53U3gJ9owb0lHbPsCich5BwFuy):  
The training code (train.py) is provided inside the `code` folder.

(3) Visualization:  
After obtaining the per-point distinctiveness, please use the provided Matlab code (inside the `visualization` folder) for shape distinction visualization.

### Questions
Please constact 'lixianzhi123@gmail.com'

