Why *survivors*?
=================

While designing and implementing the **survivors** library, we aimed to achieve the following four goals: broad functionality, applicability to real-world data, ease of use, and priority of experiments.

The broad functionality of the library's models is achieved through the continuity of their predicted functions. Discrete survival analysis models take into account the time scale before training and calculate the probability of an event occurring within a specified time. Continuous models, on the other hand, are scale-independent and can predict function values at any point in time. Additionally, the library's models allow users to predict both point values (i.e., the probability and time of an event) as well as survival and hazard functions.

The library provides an interface for building tree-based survival analysis models that do not require data preprocessing. These models are capable of handling categorical features, missing values, informative censoring, and multimodal time distributions. They also have increased sensitivity to data features, as weighted log-rank criteria are used to evaluate the significance of partitions under different time distributions. Regularization approaches are used to reduce the hazard of overfitting, making the models more suitable for real-world data.

The models are implemented using numpy, numba, and pandas, and do not require complex dependencies or heavy-weight libraries for installation. Although we did not set high-performance goals, the partition search stage makes use of parallelization of processes on the CPU, and statistical criteria are compiled into bytecode using the Numba tools. The **Survivors** has built-in source data from nine open medical datasets, which makes it easier to familiarize oneself with the functionality using real-life examples.

Note that **survivors** was designed by researchers in the field of survival analysis for conducting experimental studies. The experiment module includes many strategies for training and validating the quality of models, taking into account classical survival analysis metrics and their modifications.

Finally, let's take a look at some existing Python survival analysis libraries:

1. **Lifelines** library offers implementations of statistical models based on strict assumptions about proportional hazard, time distribution, and censoring.
2. **Pycox** extends the Cox proportional hazards model using neural networks while remaining within strict assumptions.
3. **Scikit-survival** implements classic machine learning algorithms within the scikit-learn framework, but it requires additional data preprocessing and maintains strict theoretical assumptions.

Unlike existing libraries, **survivors** allows you to work with real data "out of the box" and has increased sensitivity to data features such as time distribution and informative censoring. The implemented tree models, including survival trees and their ensembles, have been published in several papers and have shown better predictive ability compared to existing implementations.
Note that the experimental module is compatible with the models in the scikit-survival library and provides an interface for creating your forecasting models, which can be used to expand the functionality of lifelines library models.