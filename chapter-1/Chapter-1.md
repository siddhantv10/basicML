# MACHINE LEARNING BASICS NOTES
Hands on Machine Learning : Book by Aurelien Geron

>## What is Machine Learning

Machine Learning is about making machines get better at some task by learning from the data, instead of having to explicitly code rules.

>## Types of Machine Learning Systems
<!-- ------------------------------------ -->
There are so many different types of Machine Learning systems that it is useful to  classify them in broad categories, based on the following criteria:

+ Whether or not they are trained by human supervision or not:
    1. Supervised learning
    2. Unsupervised learning
    3. Semisupervised learining
    4. Reinforcement Learning

+ Whether or not they can learn incrementally on the fly or not:
    1. online learning
    2. batch learning

+ Whether they work by simply comparing new data points to known data points, or instead by detecting patterns in the training data and building a predictive model, much like scientists do:
    1. Instance based learning
    2. Model Training based learning

These criteria are not exclusive, i.e. they can be combined in any way.

## SUPERVISED/UNSUPERVISED/SEMI/REINFORCEMENT LEARNING
-----------------------------------

+ ### **Supervised Learning**

In supervised learning, the training set you feed to the algorithm includes the desired solutions, called labels.

A typical supervised learning task is **classification**. The spam filter is a good example; it is trained with many example emails along with their class (spam or ham), and it must learn how to classify new emails.

Another typical task is to predict a target numeric value, such as the price of a car, given a set of features (mileage, age, brand etc) called predictors. This sort of task is called **Regression**. To train the system, you need to give it many examples of cars, including both their predictors and their labels (i.e., their prices).

**NOTE** that some regression algorithms can be used for classification as well and vice versa. For example, logistic regression is commonly used for classification, as it can output a value that corresponds to the probability of belonging to a given class (e.g., 20% chance of being spam).

**Some of the most IMPORTANT supervised learning algorithms:**
+ k-Nearest Neighbours
+ Linear Regression
+ Logistic Regression
+ Support Vector Machines (SVMs)
+ Decision Trees and Random Forests
+ Neural Networks
-------------------------------------------------------------

+ ### **Unsupervised Learning**
<!-- ---------------------------- -->
In unsupervised learning, the training data is unlabelled. The system tries to learn without a teacher. 

**Some of the most IMPORTANT unsupervised learning algorithms:**
+ Clustering
    - K-Means
    - DBSCAN
    - Hierarchal Cluster Analysis (HCA)
+ Anomaly Detection and Novelty Detection
    - One-class SVM
    - Isolation Forest
+ Visualization and dimensionality reduction
    - Principal Component Analysis (PCA)
    - Kernel PCA
    - Locally Linear Embedding (LLE)
    - t-Distributed Stochastic Neigbor Embedding (t-SNE)
+ Association Rule learning
    - Apriori
    - Eclat

--------------------------------------
+ ### **Semi-supervised learning**

Since labeling data is usually time-consuming and costly, you will often have plenty of unlabeled instances, and few labeled instances. Some algorithms can deal with data that’s partially labeled. This is called semisupervised learning. 

Some photo-hosting services, such as Google Photos, are good examples of this. Once you upload all your family photos to the service, it automatically recognizes that the same person A shows up in photos 1, 5, and 11, while another person B shows up in photos 2, 5, and 7. This is the unsupervised part of the algorithm (clustering). Now all the system needs is for you to tell it who these people are. Just add one label per person4 and it is able to name everyone in every photo, which is useful for searching photos.

------------------------------------------

+ ### **Reinforcement Learning**

The software agent (learning system) can observe the environment, select and perform actions, and get rewards (or penalties) in return. It must then learn by itself what is the best strategy, called a policy, to get the most reward over time.

A _policy_ defines what action agent should choose when it is a given situation. 

-------------------------------

>## Main Challenges of Machine Learning

In short, since your main task is to select a learning algorithm and train it on some data, the two things that can go wrong are “bad algorithm” and “bad data.”

### **Bad Data:**

- #### Insufficient Quantity of Training Data

It takes a lot of data for most Machine Learning algorithms to work properly. Even for very simple problems you typically need thousands of examples, and for complex problems such as image or speech recognition you may need millions of examples (unless you can reuse parts of an existing model).

- #### Nonrepresentative Training Data

In order to generalize well, it is crucial that your training data be representative of the new cases you want to generalize to. This is true whether you use instance-based learning or model-based learning. 

It is crucial to use a training set that is representative of the cases you want to generalize to. This is often harder than it sounds: if the sample is too small, you will have sampling noise (i.e.,nonrepresentative data as a result of chance), but even very large samples can be nonrepresentative if the sampling method is flawed. This is called sampling bias.

- #### Poor Quality Data & Irrelevant features

If the training data is full of errors, outliers and noise, it will make it harder for the system to detect the underlying patterns, so your system is less likely to perform well. 

### **Bad Algorithm:**

- #### Overfitting the Training Data

It means that the model performs well on training data, but it doesnot generalize well. Complex models such as Deep neural networks can detect subtle patterns in data, but if the training set is noisy, or it is too small (which introduces sampling noise), then the model is likely to detect pattern in the noise iself. Obviously these patterns will not generalize to new instances. 

Overfitting happens when the model is too complex relative to the amound and noisiness of the training data. 

Possible Solutions: <br/>

1. Simplify the model by selecting one with fewer parameters (e.g. a linear model rather than a high degree polynomial model), by reducing the number of attributes in the training data, or by constraininig the model. 

2. Gather more training data

3. Reduce the noise in the training data (e.g. fix errors and remove outliers).

**Constraining the model to make it simpler and reduce the risk of overfitting is called _regularization_**

The amount of regularization to apply during learning can be controlled by a **hyperparameter** . A hyperparameter is a parameter of a learning algorithm (not of the model). As such, it is not affected by the learning algorithm itself; it must be set prior to training and remains constant during training. If you set the regularization hyperparameter to a very large value, you will get an almost flat model (a slope close to zero); the learning algorithm will almost certainly not overfit the training data, but it will be less likely to find a good solution. Tuning hyperparameters is an important part of building a Machine Learning system.

- #### Underfitting the Training Data

it occurs when your model is too simple to learn the underlying structure of the data. For example, a linear model of life satisfaction is prone to underfit; reality is just more complex than the model, so its predictions are bound to be inaccurate, even on the training examples. 

Here are the main options for fixing this problem: <br/>
1. Select a more powerful model, with more parameters.

2. Feed better features to the learning algorithm.

3. Reduce the constraints on the model.

---------------------------------------------

>## Creating Workspace

Create virtual environment using virtualenv and install required libraries. <br/>

```
cd $ML_PATH
virtualenv my_env
source my_env/bin/activate      (to deactivate type deactivate)

pip3 install numpy pandas       //install libraries you need
jupyter notebook

```

[Go to basics-1.ipynb to Continue](./basics-1.ipynb)

