
## ECG Heartbeat Classification (MIT-BIH arrhythmia)

Note: The data [MIT-BIH arrhythmia data](https://www.physionet.org/physiobank/database/mitdb/) is taken from [kaggle](https://www.kaggle.com/shayanfazeli/heartbeat).

```
Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database. IEEE Eng in Med and Biol 20(3):45-50 (May-June 2001). (PMID: 11446209) 
```

```
Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh, Mark RG, Mietus JE, Moody GB, Peng C-K, Stanley HE. PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals. Circulation 101(23):e215-e220 [Circulation Electronic Pages; http://circ.ahajournals.org/content/101/23/e215.full]; 2000 (June 13). 
```

A deep learning model based on temporal convolutional layers for the heartbeat classification was proposed in 

```
Kachuee, Mohammad, Shayan Fazeli, and Majid Sarrafzadeh. 
"ECG Heartbeat Classification: A Deep Transferable Representation." arXiv preprint arXiv:1805.00794 (2018).
```

Lets see how well we can do *without* introducing deep structures and learnable convolution parameters into a classifier. Without convolving the signals directly in the model, the signal preprocessing will have a significant impact on the performance of our models. While artificial data augmentation is performed in the original reference, we train on the raw data as given **without resampling**. We account for a sampling bias of the classes in the end.

**Spoiler**: A simple sparse benchmark GLM performs with a 85% total accuracy and a 86% total weighted recall average across the classes of the test data set. We have trouble identifying only one of the five classes with a recall of 64%.

A random Fourier feature GLM performs with a total accuracy of >90% and a total weighted recall of 92%. When accounting for the sampling bias, this adjusts to 84%.

While the CNN proposed in the original reference takes about 2 hours to train on a 1080 series GPU, the GLM can be fit on the CPU of a laptop in under a minute and still delivers feasible results when comparing to the CNN.


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```

We start off by loading the data and separating the target column from the training features.


```python
trainpath = "../data/mitbih_train.csv"
x_train = pd.read_csv(trainpath,header=None,usecols=range(187))
y_train = pd.read_csv(trainpath,header=None,usecols=[187]).iloc[:,0]
x_train.shape
```




    (87554, 187)




```python
testpath = "../data/mitbih_test.csv"
x_test = pd.read_csv(testpath,header=None,usecols=range(187))
y_test = pd.read_csv(testpath,header=None,usecols=[187]).iloc[:,0]
x_test.shape
```




    (21892, 187)




```python
x_test
```

Lets have a look at the data. We have $\sim 80k$ samples of one-beat ECG cycles. Each sample is a $[0,1]$ interval normalized timeseries padded with zeros at the end to fit a unified timeframe.


```python
def plot(x_data, y_data, classes=range(5), plots_per_class=10):

    f, ax = plt.subplots(5, sharex=True, sharey=True, figsize=(10,10))
    for i in classes:
        for j in range(plots_per_class):
            ax[i].set_title("class{}".format(i))
            ax[i].plot(x_data[y_data == i].iloc[j,:], color="blue", alpha=.5)
            
plot(x_train, y_train)
```


![png](img/output_7_0.png)



```python
def class_spec(data, classnumber, n_samples):

    fig = plt.figure(figsize=(10,13))
    if type(data)==pd.DataFrame:        
        plt.imshow(data[y_train==classnumber].iloc[:n_samples,:], 
               cmap="viridis", interpolation="nearest")
    else:
        plt.imshow(data[y_train==classnumber][:n_samples,:], 
               cmap="viridis", interpolation="nearest")
    plt.title("class{}".format(classnumber))
    plt.show()
    
for i in range(4):
    class_spec(x_train, i, 400)
```


![png](img/output_8_0.png)



![png](img/output_8_1.png)



![png](img/output_8_2.png)



![png](img/output_8_3.png)


Lets try to understand our target variable. We have a $5$-class with a heavily oversampled "0"-class. This means, that we have to account for the underrepresented classes with a weight in the loss function later on. Else, the model might classify everything a "0" and still perform well, but this is obviously not what we want.
The class encoding has the following meaning:

| class |heart condition |
|---|---|
| 0  |  Normal, Left/Right bundle branch block, Atrial escape, Nodal escape|   
| 1  | Atrial premature, Aberrant atrial premature, Nodal premature, Supra-ventricular premature  | 
| 2  | Premature ventricular contraction, Ventricular escape  |  
| 3  | Fusion of ventricular and norma  |  
| 4  | Paced, Fusion of paced and normal, Unclassifiable  |  

The distribution of the samples across the five classes:


```python
y_train.value_counts().plot(kind="bar", title="y_train")
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f5edb98c278>




![png](img/output_11_1.png)



```python
y_test.value_counts().plot(kind="bar", title="y_test")
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f5eda02d940>




![png](img/output_12_1.png)


### Preprocessing

Since we already have a lot of padded zeros in our data, it is only fair to make use of sparse matrices in the end by setting a threshold for our signal. Furthermore, we can add additional information by taking the magnitude of the dicrete signal gradients into account.

We can also try to add manual convolution, for example with a discrete gaussian. Similarly, we can incorporate manual max pooling by using the```pd.DataFrame.rolling``` function.

Additionally, we downsample our signal by using ```scipy.signal.decimate```-


```python
from scipy.signal import gaussian, decimate
from scipy.sparse import csr_matrix
```


```python
def gaussian_smoothing(data, window, std):
    gauss = gaussian(window ,std, sym=True)
    data = np.convolve(gauss/gauss.sum(), data, mode='same')
    return data

def gauss_wrapper(data):
    return gaussian_smoothing(data, 12, 7)

fig = plt.figure(figsize=(8,4))
plt.plot(x_train.iloc[1,:], label="original")
plt.plot(gauss_wrapper(x_train.iloc[1,:]), label="smoothed")
plt.legend()
```




    <matplotlib.legend.Legend at 0x7f5ed9e62f28>




![png](img/output_16_1.png)



```python
def gradient(data, normalize=True):
    data = data.diff(axis=1, periods=3)
    if normalize:
        data = data.apply(lambda x: x/x.abs().max(), axis=1)
    return data

def preprocess(data): 
    data = data.abs().rolling(7, axis=1).max()
    data = data.fillna(method="bfill",axis=1)
    #data = np.apply_along_axis(gauss_wrapper, 1, data)
    data = decimate(data, axis=1, q=5)
    data[np.abs(data) < .05] = 0
    return pd.DataFrame(data)

x_train_grad = gradient(x_train)
x_test_grad = gradient(x_test)

x_train_preprocessed = preprocess(pd.concat([x_train, x_train_grad, gradient(x_train_grad)], axis=1))
x_test_preprocessed = preprocess(pd.concat([x_test, x_test_grad, gradient(x_test_grad)], axis=1))
```


```python
plot(x_train_preprocessed, y_train)
```


![png](img/output_18_0.png)



```python
for i in range(5):
    class_spec(x_train_preprocessed, i, 200)
```


![png](img/output_19_0.png)



![png](img/output_19_1.png)



![png](img/output_19_2.png)



![png](img/output_19_3.png)



![png](img/output_19_4.png)



```python
x_train_sparse = csr_matrix(x_train_preprocessed)
```


```python
del x_train_grad
del x_test_grad
```

### Fitting a Scikit-learn benchmark sparse GLM

For a first intuition, we fit a logistic regression with a one-versus-rest approach for multilabel classification. We use the standard Newton conjugate gradient solver and add class weights according to the number of samples in the data to the loss function.

We regularize the loss with standard $L2$ penalty.


```python
import time
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report
```


```python
model = LogisticRegression(multi_class="ovr",solver="newton-cg", class_weight="balanced",
                          n_jobs=2, max_iter=150, C=.5)

start_time = time.time()
model.fit(x_train_sparse,y_train)
print("training time {}".format(time.time()-start_time))
```

    training time 19.221577405929565



```python
y_predict = model.predict(x_test_preprocessed)
cf = confusion_matrix(y_test,y_predict)
print("accuracy: " + str(accuracy_score(y_test,y_predict)))
```

    accuracy: 0.8829709482916134



```python
cf_relative = cf / cf.sum(axis=1)[:,None]
```


```python
cf
```




    array([[16139,   721,   479,   630,   149],
           [  161,   360,    27,     4,     4],
           [   71,    59,  1200,    84,    34],
           [   22,     1,    12,   127,     0],
           [   60,     9,    27,     8,  1504]])




```python
cf_relative.round(decimals=2)
```




    array([[0.89, 0.04, 0.03, 0.03, 0.01],
           [0.29, 0.65, 0.05, 0.01, 0.01],
           [0.05, 0.04, 0.83, 0.06, 0.02],
           [0.14, 0.01, 0.07, 0.78, 0.  ],
           [0.04, 0.01, 0.02, 0.  , 0.94]])




```python
print(classification_report(y_test, y_predict))
```

                 precision    recall  f1-score   support
    
            0.0       0.98      0.89      0.93     18118
            1.0       0.31      0.65      0.42       556
            2.0       0.69      0.83      0.75      1448
            3.0       0.15      0.78      0.25       162
            4.0       0.89      0.94      0.91      1608
    
    avg / total       0.93      0.88      0.90     21892
    


By the nature of our problem, we are obviously interested in high recall values for each class from "1" to "4". It is significantly worse to label a person with a relevant heart condition ("1"-"4") as normal ("0") than to label a person with a normal heart ("0") as one of the latter classes ("1"-"4").

We see that our benchmark model performs quite well for the class "4", we however misclassify a significant amount of patients in the "1" and "2" class as "0".


```python
del x_train
del x_test
del x_train_sparse
```

### Introducing nonlinearities: A random Fourier feature GLM


```python
from sklearn.kernel_approximation import RBFSampler

from scipy.stats import describe
from scipy.spatial.distance import pdist

from sklearn.mixture import GaussianMixture
```


```python
def median_heuristic(data, n_runs, n_samples):
    '''
    bootstrapping the squared euclidean distances of all data pairs to estimate the median and the mean
    -> use for bandwidth estimation
    
    https://arxiv.org/pdf/1707.07269.pdf
    '''
    medians = np.zeros(n_runs)
    means = np.zeros(n_runs)
    n_data = data.shape[0]
    sq_data = np.zeros((n_samples,n_samples))
    
    for i in range(n_runs):
        idx = np.random.randint(0, high = n_data, size = n_samples)
        sq_dist = np.triu(pdist(data.values[idx], metric="euclidean")**2)
        medians[i] = np.median(sq_dist)
        means[i] = np.mean(sq_dist)
        
    return medians, means
```


```python
medians, means = median_heuristic(x_train_preprocessed, 100, 100)
```


```python
gauss = GaussianMixture()
fit = gauss.fit(medians[:,np.newaxis])
mu = fit.means_.flatten()[0]
std = np.sqrt(fit.covariances_.flatten()[0])
print("estimated mean: {}".format(mu))
print("95% confidence: {}".format((mu-2*std/10, mu+2*std/10)))
```

    estimated mean: 0.08759222723349333
    95% confidence: (0.07737622507622226, 0.0978082293907644)



```python
gauss = GaussianMixture()
fit = gauss.fit(means[:,np.newaxis])
mu = fit.means_.flatten()[0]
std = np.sqrt(fit.covariances_.flatten()[0])
print("estimated mean: {}".format(mu))
print("95% confidence: {}".format((mu-2*std/10, mu+2*std/10)))
```

    estimated mean: 5.241464193006282
    95% confidence: (5.19404104176807, 5.288887344244493)



```python
rbf_features = RBFSampler(gamma=1/mu, n_components=550)
x_rbf = rbf_features.fit_transform(x_train_preprocessed)

for i in range(4):
    class_spec(x_rbf, i, 400)
```


![png](img/output_39_0.png)



![png](img/output_39_1.png)



![png](img/output_39_2.png)



![png](img/output_39_3.png)



```python
model = LogisticRegression(multi_class="ovr",solver="newton-cg", class_weight="balanced",
                          n_jobs=1, max_iter=150, C=.5)

start_time = time.time()
model.fit(x_rbf,y_train)
print("training time {}".format(time.time()-start_time))
```

    training time 55.57224130630493



```python
y_predict = model.predict(rbf_features.transform(x_test_preprocessed))
```


```python
print("accuracy: " + str(accuracy_score(y_test,y_predict)))
```

    accuracy: 0.9151288141786954



```python
cf = confusion_matrix(y_test, y_predict)
```


```python
cf_relative = cf / cf.sum(axis=1)[:,None]
print(cf_relative.round(decimals=2))
```

    [[0.92 0.03 0.02 0.02 0.  ]
     [0.23 0.71 0.04 0.01 0.01]
     [0.04 0.02 0.88 0.05 0.01]
     [0.09 0.01 0.04 0.87 0.  ]
     [0.03 0.   0.02 0.   0.94]]



```python
print(classification_report(y_test, y_predict))
```

                 precision    recall  f1-score   support
    
            0.0       0.99      0.92      0.95     18118
            1.0       0.40      0.71      0.51       556
            2.0       0.74      0.88      0.81      1448
            3.0       0.23      0.87      0.37       162
            4.0       0.93      0.94      0.94      1608
    
    avg / total       0.95      0.92      0.93     21892
    


### Unbiased model evaluation


```python
y_test_list = [y_test[y_test==i][:150] for i in range(5)]
y_test_mod = np.hstack(y_test_list)
x_test_list = [rbf_features.transform(x_test_preprocessed)[y_test==i][:150,:] for i in range(5)]
x_test_mod = np.vstack(x_test_list)
```


```python
y_predict = model.predict(x_test_mod)
print("accuracy: " + str(accuracy_score(y_test_mod,y_predict)) + "\n")
cf = confusion_matrix(y_test_mod, y_predict)
cf_relative = cf / cf.sum(axis=1)[:,None]
print(cf_relative.round(decimals=2))
print(classification_report(y_test_mod, y_predict))
```

    accuracy: 0.844
    
    [[0.89 0.04 0.03 0.03 0.01]
     [0.23 0.71 0.04 0.01 0.01]
     [0.05 0.02 0.85 0.06 0.03]
     [0.08 0.01 0.03 0.88 0.  ]
     [0.07 0.01 0.03 0.01 0.89]]
                 precision    recall  f1-score   support
    
            0.0       0.68      0.89      0.77       150
            1.0       0.91      0.71      0.80       150
            2.0       0.86      0.85      0.86       150
            3.0       0.89      0.88      0.88       150
            4.0       0.96      0.89      0.92       150
    
    avg / total       0.86      0.84      0.85       750
    

