# Find patterns in data which are interesting with respect to background knowledge represented by a Bayesian network

This software is based on the following papers:

<a id="1">[1]</a> S. Jaroszewicz, D. Simovici. *Interestingness of Frequent Itemsets Using Bayesian Networks as Background Knowledge*. In 10th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD 2004), pages 178-186, Seattle, WA, August, 2004. 

<a id="2">[2]</a> S. Jaroszewicz, T. Scheffer. *Fast Discovery of Unexpected Patterns in Data, Relative to a Bayesian Network*. In 11th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD 2005), pages 118-127, Chicago, IL, August, 2005.

<a id="3">[3]</a> S. Jaroszewicz, T. Scheffer, D.A. Simovici. *Scalable pattern mining with Bayesian networks as background knowledge*. Data Mining and Knowledge Discovery, 18(1), pages 56-100, 2009.

## What you need:
* Python 3.x
* the numpy package

## How to run the code:
* unpack the source
* change to the top level source directory
* type `python BNInterGUI.py`

## Example session
After you type `python BNInterGUI.py` two windows will appear:

![Network window](assets/images/start_network.png)
![Patterns window](assets/images/start_patterns.png)

the left window show the Bayesian network describing current
background knowledge, the right window displays interesting patterns.
Currently both windows are empty.

Let us now load an example dataset.  Click the `Browse` button next to
`Data file` field in the patterns window and select the
`data/ksl_discr.arff` file.   The Bayesian network window should now show a network with no edges:

![Empty Bayesian network](assets/images/network_empty.png)

meaning that to all variables assumed to be independent.

Let us now briefly describe the dataset which comes from the `deal' R
package.  The dataset contains data on elderly Danes collected in 1967
and 1984.  All numerical variables are discretized into three levels
0,1,2.  Full variable list:

* **FEV**  Forced Ejection Volume (lung capacity)
* **Kol**  Cholesterol
* **Hyp**  Hypertension {0,1}
* **BMI**  Body Mass Index
* **Smok**  Smoking
* **Alc**  Alcohol consumption
* **Work**  Does the subject work
* **Sex**  Male or Female
* **Year**  Year of the study 1967 or 1984

Let us now discover interesting patterns.  In the pattern window you
can select the algorithm to use (sampling based or exact) as well as
its parameters:

* `max pattern size` maximum size of the discovered pattern
* `min inter` minimum interestingness level of patterns which is used
  for pruning.  Expressed as the minimum number of records of
  difference in pattern freqeuncy in data and based on Bayesian
  network prediction.
* `# best patterns to find` only affects the sampling based algorithm.
  More patterns are actually found, but this number is guaranteed to
  be statistically correct.
* `error prob.` upper bound probability that one of the top patterns
  is incorrect (only affects the sampling based algorithm).

Select the `exact` algorithm (faster for tiny data) and click the
`Run` button.

You can now load a Bayesian network representing background knowledge.
Click the `Browse` button next to `Bayes net` field in the patterns
window and select the `data/ksl_discr.net` file.  The background
knowledge comes from [[1]](#1) and represents the authors' common sense
knowledge.

The following network should appear:

![Background knowledge](assets/images/network_paper.png)


