# KFS-TUNE: Kernel-based Feature Selection for efficiency and
## Kernel-based Feature Selection for efficiency and accuracy tuning in Time Series Classification


The present repository is dedicated to Time Series Classification. It contains the implementations of the following:

### 1. Creation of a new convolutional algorithm that uses feature selection **(KFS-TUNE)**
  We introduce KFS-TUNE, which achieves an incredible balance between accuracy and time. Our approach exhibits the highest performance in terms of time complexity, due to the use of random kernels in
  combination with feature selection - a method which we first introduce in combination with convolutional classifiers - especially when taking into consideration that time series
  classifiers, especially deep-leaning models, possess increased time complexities.

  We introduce KFS-TUNE's flow below:
![image](https://github.com/user-attachments/assets/dcd74a1f-e6ee-4be0-ad39-fed48ecde109)



### 1. Systematic evaluation on various datasets:
#### UCR Archive
   The datasets are organized based on dimensions (univariate or multivarite)
      * Length (<300, >=300, >700)
         * Classes (<10, >=10, >=30)
     
   We conducted our experiments on various datasets for this part, but we include here 11, since based on the former categorization UCR contains this number of datasets.
   We use numerous datasets to evaluate 21 classifiers which we separate in 6 categories:
   1. Deep Learning:
      * Multi Layer Perceptron (MLP)
      * Convolutional Neural Network (CNN)
      * Fully Convolutional Network (FCN)
      * Multi-Channels Deep Convolutional Neural Networks (MCD-CNN)
   2. Dictionary - based:
      * Bag of SFA Symbols (BOSS)
      * Contractable Bag of SFA Symbols (cBOSS)
      * Individual Bag of SFA Symbols (iBOSS)
      * Temporal Dictionary Ensemble (TDE)
      * Individual Temporal Dictionary Ensemble (iTDE)
      * Word Extraction for Time Series Classification (WEASEL)
      * MUltivariate Symbolic Extension (MUSE)
   3. Distance - based:
      * Shape Dynamic Time Warping (shapeDTW)
      * K-Nearest Neighbors (using DTW) (KNN)
   4. Feature - based:
      * Canonical Time-series Characteristics (catch22)
      * Fresh Pipeline with RotatIoN forest (FreshPRINCE)
   5. Interval - based:
      * Supervised Time Series Forest (STSF)
      * Time Series Forest (TSF)
      * Canonical Interval Forest (CIF)
      * Diverse Representation Canonical Interval Forest (DrCIF)
   6. Convolution - based:
      * ROCKET
      * Arsenal

  We are using aeon library while except for MCD-CNN which is only available on sktime.
  For the evaluation we create metrics such as F1, Accuracy, Precision, AUC scores (macro and micro) while we also calculate execution times and memory consumptions.
  We also create plots such as macro average ROC AUC curve per classifier, confusion matrices and ROC AUC curves for each class.
  We also introduce results with and without cross-validation. For cross-validation we use both k-fold and TimeSeriesSplit. We use k-fold only for comparison reasons with other papers,
  since we are opposite to its use for time series data, for temporal structure reasons. Therefore, we always suggest TimeSeriesSplit if you want to use cross-validation.

#### LifeSnaps dataset
LifeSnaps dataset, a multi-modal, longitudinal, and geographically-distributed dataset containing a plethora of anthropological data, collected unobtrusively for the total course of more than 4 months by n = 71 participants. LifeSnaps contains more than 35 different data types from second to daily granularity, totaling more than 71 M rows of data.
The dataset can be found here: [LifeSnaps - Zenodo](https://www.nature.com/articles/s41597-022-01764-x)


#### Synthetic extreme datasets
The synthetic datasets are critical for controlled experimentation and stress testing. They introduce diverse temporal patterns, overlaps, and noise levels, mimicking even extreme real-world complexities. They also enable the exploration of scalability and robustness in handling large datasets and assessing the algorithm's ability to capture patterns across different temporal scales.
Specifically, The synthetic datasets were meticulously generated to mimic very extreme and complex time series data with a variety of characteristics. Each dataset comprises multiple time series samples, where the number of samples and their lengths were randomly selected within defined ranges to introduce variability. Specifically, the length of individual time series was chosen randomly between 30 and 1000 points, while the total number of time series in a dataset ranged from 100 to 1000, ensuring datasets of diverse sizes. Each time series was labeled with one of three classes (0, 1, or 2), where the probability of a class determined the pattern and behavior of the series. For instance, label 0 had a higher likelihood of generating a simple sine wave pattern ('a'), whereas label 1 predominantly generated a modulated sine wave pattern ('b').

The generated time series displayed distinct patterns based on the assigned label. Pattern 'a' consisted of a standard sine wave with additive Gaussian noise, simulating minor natural variability. Pattern 'b' featured a modulated sine wave with doubled frequency and an additional layer of Gaussian noise. For certain labels, pattern 'b' also included spikes in 5% of the data points, simulating random, extreme deviations to mimic real-world anomalies. Lastly, pattern 'c' combined two sine waves with different frequencies, augmented by a quadratic trend that depended on the label, and included low-level Gaussian noise for further complexity.

Additionally, a single very large dataset was generated to simulate a massive time series classification problem. This dataset consisted of 100,000 time series, each with a length randomly selected between 30 and 70 points. The large dataset retained the same class distribution and noise characteristics as the smaller datasets, ensuring consistency in the patterns ('a', 'b', and 'c') across the samples. However, the sheer size of the dataset posed a greater computational challenge, testing the scalability of classification algorithms. Each series in the dataset was generated with label-specific behaviors and noise profiles, as described previously, with Gaussian noise added based on the series' assigned label and pattern type. This dataset also allowed for analysis of the algorithm’s performance under high-volume conditions, providing insights into training time, computational efficiency, and accuracy when handling large-scale time series data.
