.. -*- mode: rst -*-

|Price| |License| |PyPi| |DOI|

.. |Price| image:: https://img.shields.io/badge/price-FREE-0098f7.svg
   :target: https://github.com/iuliivasilev/dev-survivors/blob/master/LICENSE

.. |PyPi| image:: https://img.shields.io/pypi/v/survivors
    :target: https://pypi.org/project/survivors/

.. |License| image:: https://img.shields.io/badge/license-BSD%203--Clause-blue.svg
   :target: https://github.com/iuliivasilev/dev-survivors/blob/master/LICENSE

.. |DOI| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.10649986.svg
    :target: https://zenodo.org/doi/10.5281/zenodo.10649777

=========
survivors
=========

Event analysis has many applications: healthcare, hardware, social science, bioinformatics, and more. Survival analysis allows you to predict not only the time and probability of an event but also how the probability of that event changes over time.
In particular, there are three functions: the survival function *S(t)*, the density function *f(t)*, and the hazard function *h(t)*:

.. math::
    S(t)=P(T>t), f(t)=(1 - S(t))', h(t)=f(t)/S(t)

The open-source **survivors** library aims to fit accurate data-sensitive tree-based models. 
These models handle categorical features and deal with missing values,overcoming the limitations of existing `lifelines <https://github.com/lifelines/lifelines?ysclid=lta0m13i2b832399887>`_, `scikit-survival <https://github.com/sebp/scikit-survival>`_, and `pycox <https://github.com/havakv/pycox>`_ models.
Survivors is a platform for conducting experimental research. The experiment module is compatible with scikit-survival and lifelines models (non-parametric and parametric models have already been embedded into the library).

Developed models published in scientific articles [1]_, [2]_, [3]_ and outperformed existing models in terms of accuracy based on open medical data. We invite survival analysis researchers to join the development of survivors, using the library for their projects, reporting any problems, and creating new solutions.
Documentation is available on https://iuliivasilev.github.io/dev-survivors/

Principles
-----------

Built-in **survivors** models were developed as part of a PhD thesis at Lomonosov Moscow State University. The goal of the library is to analyze existing methods of survival analysis and develop new techniques to overcome these limitations.

Existing methods are not suitable for real-world data. Discrete methods use a fixed time scale. Statistical methods are based on strong assumptions, and tree-based methods use the log-rank statistic with low sensitivity.
**Survivors** has the following features:

1. Continuous Predictions: The timeline is user-friendly and only needs to be set at the prediction stage.
2. Modified Quality Metrics: Existing metrics are excessively sensitive to data features, such as class imbalance, event distribution, and timeline.
3. Weighted Survival Tree. For the first time, CRAID uses weighted log-rank criteria. Wilcoxon, Peto, and Tarone-Ware weights increase the significance of events within a certain time interval.
4. Speed of work. The models are developed from scratch and utilize parallelization, vectorization, and JIT compilation. A new histogram-based method is employed to identify splits in censored data. This method optimizes memory usage and has a high level of operation speed.
5. Ease of Use. CRAID and ensembles work out-of-the-box. Categorical and missing data are processed within the models.
6. A Platform for Experiments. The experiments module provides a flexible interface for working with built-in and external survival models, their hyperparameters, and experiment strategies, such as hold-out, cross-validation, grid search with cross-validation and sampling, and time-aware cross-validation.

Installation
------------

User Installation
~~~~~~~~~~~~~~~~~

The most convenient and fastest way to install a package is to directly download the library from the Python package catalog (Python Package Index, PyPI).
The version of the source files in the directory is up-to-date and consistent with the GitHub repository::

  pip install survivors

An alternative installation method is based on the use of source files. 
The first step is to download the source files using Github::

  git clone command https://github.com/iuliivasilev/dev-survivors.git

Or getting and unpacking the archive of `the latest published version <https://github.com/iuliivasilev/dev-survivors/releases/>`_. Next, use the command line to go to the **dev-survivors** directory. Finally, the manual installation of the library is completed after executing the following command::

  python command setup.py install

Requirements
~~~~~~~~~~~~

- Python (>= 3.9)
- NumPy (>= 1.22)
- Pandas (>=0.25)
- Numba (>= 0.58.0)
- matplotlib (>= 3.5.0)
- seaborn
- graphviz (>= 2.50.0)
- joblib
- scikit-learn (>= 1.0.2)
- openpyxl

Optional for comprehensive experiments:

- lifelines (>= 0.27.8)
- scikit-survival (>= 0.17.2)

Examples
------------

The user guides in the *doc* and *demonstration* directories provide detailed information on the key concepts for **survivors**. 
They also include hands-on examples in the form of `Jupyter notebooks <https://jupyter.org/>`_.
In particular, the library allows users to carry out a range of scenarios.

1. Loading and preparing 9 open medical datasets: GBSG, PBC, SMARTO, SUPPORT2, WUHAN, ACTG, FLCHAIN, ROTT2, FRAMINGHAM.
2. Fitting Survival Analysis Models: There are the following models available: a Decision Tree (CRAID), a Bootstrap Ensemble (BootstrapCRAID), and an Adaptive Boosting Ensemble (BoostingCRAID). Each model has a wide range of hyperparameters, providing flexibility for the model.
3. Predict the probability and timing of the event. Forecasts can help users solve the problem of classifying or ranking new patients based on the expected severity of their disease. 
4. Predict the individual survival functions and cumulative hazards of patients. Forecasts can be used to support medical decisions and adjust treatments.
5. Visualizing and interpreting dependencies in data.

Help and Support
----------------

Communication
~~~~~~~~~~~~~

- Email: iuliivasilev@gmail.com
- LinkedIn: https://www.linkedin.com/in/iulii-vasilev


Citation
~~~~~~~~~~

If you use **survivors** in a scientific publication, we would appreciate citations:

.. [1] Vasilev I., Petrovskiy M., Mashechkin I. Survival Analysis Algorithms based on Decision Trees with Weighted Log-rank Criteria. - 2022.

.. [2] Vasilev, Iulii, Mikhail Petrovskiy, and Igor Mashechkin. "Sensitivity of Survival Analysis Metrics." Mathematics 11.20 (2023): 4246.

.. [3] Vasilev, Iulii, Mikhail Petrovskiy, and Igor Mashechkin. "Adaptive Sampling for Weighted Log-Rank Survival Trees Boosting." International Conference on Pattern Recognition Applications and Methods. Cham: Springer International Publishing, 2021.

.. _survival analysis: https://en.wikipedia.org/wiki/Survival_analysis
