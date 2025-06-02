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

Key Features
------------

Survivors was developed during a PhD project at Lomonosov Moscow State University to overcome the limitations of traditional survival analysis.
It is designed to handle data with informative censoring and arbitrary time-to-event distributions. Here's what makes it stand out:

* **Data Included** – Comes with 20+ real-world `datasets <https://iuliivasilev.github.io/dev-survivors/modules/Datasets.html>`_ from healthcare, CRM, and reliability.
* **True Continuous-Time Modeling** – No discretization needed; time is continuous where it matters: at prediction.
* **Custom Metrics That Matter** – Over 10 evaluation metrics crafted to handle censoring, imbalance, and rare events.
* **Speed Meets Scale** – Optimized from scratch with JIT compilation, vectorized math, and histogram-based splits for efficient performance.
* **Experiment-Ready** – Built-in module for reproducible experiments: hold-out, cross-validation, time-aware splits, and hyperparameter search.
* **Meta-Modeling Power** – Create stratified models and ensemble aggregators with ease.
* **Broad Applicability** – Suitable for a wide range of domains including healthcare (death or discharge prediction), industrial maintenance (equipment failure), and business analytics (customer churn and retention).

Installation
------------
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

Getting Started
---------------

The `user guides <https://iuliivasilev.github.io/dev-survivors/auto_examples/plot_user_guide.html>`_ provide detailed information on the key concepts for **survivors**::

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

* **CRAID** – interpretable decision tree model for survival analysis. [1]_
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