
===============
survivors
===============

survivors is a Python module for `survival analysis`_. 

It allows for building survival models: Survival Tree, Bagging ensemble, Adaptive Boosting ensemble.

=======================
About Survival Analysis
=======================

The objective in `survival analysis`_ (also referred to as time-to-event or reliability analysis)
is to establish a connection between covariates and the time of an event.

Survival analysis is a set of statistical models and methods used for estimating time until the occurrence of an event (or the probability that an event has not occurred). These methods are widely used in demography, e.g. for estimating lifespan or age at the first childbirth, in healthcare, e.g. for estimating the duration of staying in a hospital or survival time after the diagnosis of a disease, in engineering (for reliability analysis), in insurance, economics, and social sciences.

Statistical methods need data, but complete data may not be available, i.e. the exact time of the event may be unknown for certain reasons  (the event did not occur before the end of the study or it is unknown whether it occurred). In this case, events are called censored. The data are censored from below (left-censored) when below a given value the exact values of observations are unknown. Right censored data (censored from above) does not have exact observations above a given value. Further in this paper, right censoring is considered.

============
Requirements
============

- Python 3.7 or later
- joblib
- pickle-mixin
- numpy
- numba
- matplotlib
- seaborn
- graphviz
- pandas >=0.25
- scipy
- python-dateutil
- scikit-learn
- lifelines
- scikit-survival
- openpyxl

============
Installation
============

The easiest way to install survivors is to use by running::

  pip install survivors

========
Examples
========

The `user guide <https://google.com>`_ provides
in-depth information on the key concepts of survivors, an overview of available survival models,
and hands-on examples in the form of `Jupyter notebooks <https://jupyter.org/>`_.

The "demonstration" directory contains examples of using the library as Jupyter Notebook.
The example is a variant of solving a multi-stage problem:

1. Import the library
2. Loading and preparing data
3. Model training
4. Getting a forecast
5. Visualization of the constructed model
6. Visualization of true and predicted survival functions

==========
Scenarios
==========

Based on the **survivors**, it is possible to carry out the following work scenarios:

1. Collecting data from patient medical histories (hospital tests, medications, treatment) from various medical institutions. The medical history can be presented as a set of tables in csv or xlsx format, or as a hierarchical structure of xml files.

2. Building survival analysis models. There are available the following models: a decision tree, a bagging ensemble, and a boosting ensemble. For each model, there is a wide range of hyperparameters, which provide the flexibility of the model.

3. Predicting the probability and time of the event. Forecasts can be used by the user to solve the problem of classifying or ranking new patients according to the expected severity of the disease.

4. Predicting the individual survival functions and cumulative hazard of patients. Forecasts can be used to support medical decisions and adjust treatment.


==========
References
==========

Methods from **survivors** are based on following paper.

Vasilev I., Petrovskiy M., Mashechkin I. Survival Analysis Algorithms based on Decision Trees with Weighted Log-rank Criteria. - 2022.

.. code::

@inproceedings{vasilev2022survival,
    title={Survival Analysis Algorithms based on Decision Trees with Weighted Log-rank Criteria.},
    author={Vasilev, Iulii and Petrovskiy, Mikhail and Mashechkin, Igor V},
    booktitle={ICPRAM},
    pages={132--140},
    year={2022}
}

@inproceedings{vasilev2023adaptive,
    title={Adaptive Sampling for Weighted Log-Rank Survival Trees Boosting},
    author={Vasilev, Iulii and Petrovskiy, Mikhail and Mashechkin, Igor},
    booktitle={Pattern Recognition Applications and Methods: 10th International Conference, ICPRAM 2021, and 11th International Conference, ICPRAM 2022, Virtual Event, February 4--6, 2021 and February 3--5, 2022, Revised Selected Papers},
    pages={98--115},
    year={2023},
    organization={Springer}
}

@article{vasilev2023sensitivity,
    title={Sensitivity of Survival Analysis Metrics},
    author={Vasilev, Iulii and Petrovskiy, Mikhail and Mashechkin, Igor},
    journal={Mathematics},
    volume={11},
    number={20},
    pages={4246},
    year={2023},
    publisher={MDPI}
}

.. _survival analysis: https://en.wikipedia.org/wiki/Survival_analysis