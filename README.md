# Evolutionary Federated Learning on EEG-data

This repository contains the code for the publication [Evolutionary Federated Learning on EEG-data](https://www.semanticscholar.org/paper/Evolutionary-Federated-Learning-on-EEG-data-Szegedi-Kiss/843476cfaed7afa24a7646e51379f12ad29c4ba2).

### Abstract

In the paper we try to train a Convolutional Neural Network to solve the [UCI EEG Classification problem](http://archive.ics.uci.edu/ml/datasets/EEG+Database).


The novelty of our method lies in modifying [Google's Federated Learning](https://ai.googleblog.com/2017/04/federated-learning-collaborative.html)
method using Evolutionary Algorithm for added security. 


The added value is that the sensitive medical data can reside on the federated nodes as the model evaluation happens on the nodes locally and 
the results are aggregated on the server side. 

### The codebase

Here we lay out a high level guide on what the most important files do:

- **baseline_shallow.ipynb**: Here we design and train the Neural Network using the usual SGD method. This is what we use as baseline. 
- **common.py**: Utility functions for loading and representing the data
- **node.py**: Node representation classes
- **genetic.py**: This is the core framework for running the Federated Neuroevolution training with different paremeters
- **federated_genetic_*.py**: These files are concrete runs with different hyperparameters for Federated Neuroevolution

The other notebook files are early proof of concepts and visualizations.

### Citations

**Please if you use our code do not forget to properly cite our publication. 
Bibtex citation:**

```tex
@article{szegedi2019evolutionary,
  title={Evolutionary federated learning on EEG-data},
  author={Szegedi, G{\'a}bor and Kiss, P{\'e}ter and Horv{\'a}th, Tom{\'a}{\v{s}}},
  year={2019}
}
```

