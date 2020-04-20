# Unsupervised Detection of Distinctive Regions on 3D Shapes
by [Xianzhi Li](https://nini-lxz.github.io/), [Lequan Yu](https://yulequan.github.io/), [Chi-Wing Fu](https://www.cse.cuhk.edu.hk/~cwfu/), [Daniel Cohen-Or](https://www.cs.tau.ac.il/~dcor/) and [Pheng-Ann Heng](http://www.cse.cuhk.edu.hk/~pheng/).

### Introduction
This repository is for our ACM Transactions on Graphics (TOG) 2020 paper '[Unsupervised Detection of Distinctive Regions on 3D Shapes](https://arxiv.org/abs/1905.01684)'. In this paper, we present presents a novel approach to learn and detect distinctive regions on 3D shapes. Unlike previous works, which require labeled data, our method is unsupervised. We conduct the analysis on point sets sampled from 3D shapes, then formulate and train a deep neural network for an unsupervised shape clustering task to learn local and global features for distinguishing shapes with respect to a given shape set. To drive the network to learn in an unsupervised manner, we design a clustering-based nonparametric softmax classifier with an iterative re-clustering of shapes, and an adapted contrastive loss for enhancing the feature embedding quality and stabilizing the learning process. By then, we encourage the network to learn the point distinctiveness on the input shapes. We extensively evaluate various aspects of our approach and present its applications for distinctiveness-guided shape retrieval, sampling, and view selection in 3D scenes.

In this repository, we shall release code and data soon. 

### Questions
Please constact 'lixianzhi123@gmail.com'

