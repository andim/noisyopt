---
title: 'Noisyopt: A Python library for optimizing noisy functions.'
tags:
  - optimization
  - stochastic approximation
  - spsa
authors:
 - name: Andreas Mayer
   orcid: 
   affiliation: 1
affiliations:
 - name: Laboratoire de Physique Théorique, École Normale Supérieure
   index: 1
date: 30 March 2017
bibliography: paper.bib
---

# Summary

Optimization problems have great practical importance across many fields. Sometimes a precise evaluation of the function to be optimized is either impossible or exceedingly computationally expensive. An example of the former case is optimization based on a complex simulation, an example of the latter arises in machine learning where evaluating a loss function over the complete data set can be to expensive. Optimization based on noisy evaluations of a function is thus an important problem.

Scipy.optimize [@scipy] provides simple-to-use implementations of optimization algorithms for generic optimization problem. Noisyopt provides algorithms taylored to noisy optimization problems with a compatible call syntax. It implements adaptive pattern search and simultaneous perturbation stochastic approximation. Bound constraints on variables are supported. The library furthermore has methods for finding a root of a noisy function by an adaptive bisection algorithm.

# References
