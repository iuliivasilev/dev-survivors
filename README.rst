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

.. image:: https://github.com/iuliivasilev/dev-survivors/blob/master/docs/static/collage.png
  :target: https://iuliivasilev.github.io/dev-survivors/

**Survivors** is a Python library for survival analysis designed to handle real-world, messy, and heterogeneous data with high accuracy and interpretability.

Unlike many existing libraries that rely on strong theoretical assumptions (e.g., proportional hazards, exponential distributions), **Survivors** is:

* **Flexible**: works with incomplete, noisy, and heterogeneous tabular data.
* **Interpretable**: powered by survival trees and ensembles (forests) that offer transparent decision paths.
* **Highly sensitive to data structure**: tree splits are adapted to handle informative censoring and arbitrary time-to-event distributions.
* **Accurate**: avoids restrictive model assumptions, resulting in higher predictive performance in complex settings.


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

Install the latest release via pip (`PyPI <https://pypi.org/project/survivors/>`_)::

    pip install survivors

Or install the development version::

    git clone https://github.com/iuliivasilev/dev-survivors.git
    cd dev-survivors
    pip install -e .

Dependencies
~~~~~~~~~~~~

* Python >= 3.8
* NumPy, Pandas, Numba, Scikit-learn, Matplotlib

Optional for comprehensive experiments:

* lifelines (>= 0.27.8)
* scikit-survival (>= 0.17.2)

Quickstart
------------

The user guides in the *doc* and *demonstration* directories provide detailed information on the key concepts for **survivors**::

    import survivors.datasets as ds
    import survivors.constants as cnt
    from survivors.tree import CRAID

    X, y, features, categ, sch_nan = ds.load_gbsg_dataset()

    cr = CRAID(criterion='wilcoxon', depth=2, min_samples_leaf=0.1, 
            signif=0.05, categ=categ, leaf_model="base")
    cr.fit(X, y)
    tree_view = cr.visualize(mode="surv")

    tree_view

.. image:: https://github.com/iuliivasilev/dev-survivors/blob/master/demonstration/CRAID_GBSG_depth2.png

Supported Models
----------------

* **CRAID** – interpretable decision tree model for survival analysis [1]_
* **ParallelBootstrapCRAID** – ensemble of independent trees with improved stability and accuracy [2]_
* **BoostingCRAID** (optional) – boosting ensemble with adaptive sampling [3]_
* **Modified existing models** (Kaplan-Meier, CoxPH, AFT ...)

Help and Support
----------------

We welcome contributions!
Open issues, propose features, and submit pull requests via GitHub.

For questions, discussions, or collaboration ideas, feel free to open an issue or `contact the maintainer directly <https://www.linkedin.com/in/iulii-vasilev>`_ (`email <iuliivasilev@gmail.com>`_).

Citation
~~~~~~~~~~

If you use **survivors** in a scientific publication, we would appreciate citations:

.. [1] Vasilev I., Petrovskiy M., Mashechkin I. Survival Analysis Algorithms based on Decision Trees with Weighted Log-rank Criteria. - 2022.

.. [2] Vasilev, Iulii, Mikhail Petrovskiy, and Igor Mashechkin. "Sensitivity of Survival Analysis Metrics." Mathematics 11.20 (2023): 4246.

.. [3] Vasilev, Iulii, Mikhail Petrovskiy, and Igor Mashechkin. "Adaptive Sampling for Weighted Log-Rank Survival Trees Boosting." International Conference on Pattern Recognition Applications and Methods. Cham: Springer International Publishing, 2021.

.. _survival analysis: https://en.wikipedia.org/wiki/Survival_analysis