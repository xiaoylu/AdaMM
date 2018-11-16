# Adaptive Modularity Maximization

Similarity measure between data points plays an important role in clustering. In network science, the community detection algorithms can be also significantly improved by using proper edge weighting scheme. The focus of our study is how to build a novel model which assigns such optimal weights to edges.

Our model learns such edge weights in a supervised manner. Provided a set of group truth communities, we require our model to automatically learn the edge weighting scheme. This is probably the most unique feature of our work. Once the parameters are inferred, we can apply this regression model to assign weights to edges in any graph, considering the comprehesive definition of local community structures in various network instances.

In our model, the edge weights are designed to justify the local optimality of the modularity of the group truth communities. Modularity, proposed by Newman et al., is a widely used quality measure to evaluate communities. A good partition of the network usually leads to high modularity. For this reason, many modularity maximazation algorithms, as their names suggested, finds the communities in a network to maximize the modularity. However, all of them suffer from the resolution limit problem which refers to the bias toward large communities over small ones. Our work shows that proper edge weights can conquer this issue and thus leads to desirable community detection results.

We have provided the Python code for the training of our regression model on github. By running:
```python
python main.py
```
to test the performance on American college football network

Please consider citing our paper if you find the code useful
[Adaptive modularity maximization via edge weighting scheme](https://www.sciencedirect.com/science/article/pii/S0020025517301068)
```
@article{lu2018adaptive,
  title={Adaptive modularity maximization via edge weighting scheme},
  author={Lu, Xiaoyan and Kuzmin, Konstantin and Chen, Mingming and Szymanski, Boleslaw K},
  journal={Information Sciences},
  volume={424},
  pages={55--68},
  year={2018},
  publisher={Elsevier}
}
```
