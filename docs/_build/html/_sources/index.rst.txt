.. survivors documentation master file, created by
   sphinx-quickstart on Thu Feb 29 23:54:19 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


=========
survivors
=========

`survivors <https://pypi.org/project/survivors/>`_ is a Python library for **survival analysis**.

*survivors* contains multiple modern tree-based survival models: Survival Tree, Bagging ensemble, Adaptive Boosting ensemble.

The library allows to carry out the following scenarios:

1. Collect data from patient medical histories (hospital tests, medications, treatment) from various medical institutions. The medical history can be presented as a set of tables in csv or xlsx format, or as a hierarchical structure of xml files.
2. Build survival analysis models. There are available the following models: a decision tree, a bagging ensemble, and a boosting ensemble. For each model, there is a wide range of hyperparameters, which provide the flexibility of the model.
3. Predict the probability and time of the event. Forecasts can be used by the user to solve the problem of classifying or ranking new patients according to the expected severity of the disease.
4. Predict the individual survival functions and cumulative hazard of patients. Forecasts can be used to support medical decisions and adjust treatment.
5. Run experiments by one of multiple scheme (HOLD-OUT, CV, CV+SAMPLE, Time-based CV).
6. Evaluate the quality of models and compare its on different survival analysis metrics: CI, IBS, IAUC, AUPRC, LIKELIHOOD.

Documentation
------------------------------

.. toctree::
   :maxdepth: 1
   :caption: Architecture

   Component description
   API Reference


.. toctree::
  :maxdepth: 1
  :caption: About survivors

  About survival analysis <AboutSA>
  Why survivors? <Why>
  Installation
  Citing survivors <Citing>

Indices and tables
------------------------------

* :ref:`sphx_glr_auto_examples_plot_user_guide.py`
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
