# Matrix-Factorization-for-Recommendation
Using Matrix Factorization/Probabilistic Matrix Factorization to solve Recommendation.
我使用R语言实现了三种矩阵分解算法：矩阵分解（利用动量优化算法）（mf），概率矩阵分解（利用动量优化算法）（pmf）以及概率矩阵分解（利用随机梯度下降优化算法）（pmf-sgd）  
我使用了三种推荐系统常见的数据集：Epinions，Movielens(100k)，Netflix(1M)  
我使用的评价指标包括了均方根误差(RMSE)与绝对平均误差(MAE)，精确率Pre@K(i)与召回率Re@K(i)  
****
![Image text](https://github.com/stxupengyu/Matrix-Factorization-for-Recommendation/blob/master/img-folder/1.png)  
****
I use R to implement three Matrix Factorization algorithms: Matrix Factorization (using momentum optimization algorithm) (MF), probability Matrix Factorization (using momentum optimization algorithm) (PMF) and probability Matrix Factorization (using random gradient descent optimization algorithm) (PMF-SGD)  
I used three common datasets for recommendation systems: epinations, movies (100k), Netflix (1m)  
The evaluation indexes I used include root mean square error (RMSE) and absolute mean error (MAE), Precision Pre@K(i) and Recall Re@K(i)  
****
Reference:  
[1] Y. Koren. Factorization meets the neighborhood: Amultifaceted collaborative ﬁltering model. In Proceeding of the 14th ACM SIGKDD international conferenceon Knowledge discovery and data mining, 2008.  
[2] Y. Koren. Collaborative ﬁltering with temporal dynamics. In KDD-09, 2009.  
[3] R. Salakhutdinov and A. Mnih. Probabilistic matrixfactorization. In Advances in Neural Information Processing Systems (NIPS), volume 20, 2007.  
[4] R. Salakhutdinov and A. Mnih. Bayesian probabilistic matrix factorization using markov chain monte carlo.In Proceedings of the Twenty-Fifth International Conference on Machine Learning (ICML 2008), Helsinki,Finland, 2008.  
[5] Yehuda Koren, Robert Bell, and Chris Volinsky. 2009. Matrix factorization techniques for recommender systems.Computer 42, 8 (2009), 30–37.  
[6]Huafeng Liu, Liping Jing, Yuhua Qian, and Jian Yu. 2019. Adaptive Local Low-rank Matrix Approximation for Recommendation. ACM Transactions on Information Systems *, *, Article * (June 2019), 32 pages.   
[7]http://www.trustlet.org/downloaded_epinions.html  
[8]https://grouplens.org/datasets/movielens/  
[9]https://www.netflixprize.com  
[10] Ruder S. An overview of gradient descent optimization algorithms[J]. arXiv preprint arXiv:1609.04747, 2016.  
[11] http://www.utstat.toronto.edu/~rsalakhu/BPMF.html  
