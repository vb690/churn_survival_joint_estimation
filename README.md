# Modelling Early User-Game Interactions for Joint Estimation of Survival Time and Churn Probability
Repository hosting a minimal version of the code employed for the paper:  
  
`Modelling Early User-Game Interactions for Joint Estimation of Survival Time and Churn Probability` 
  
# Overview
The aim of our work was to develop a more generalizable and holistic methodlogy for modelling user engagement which possessed characteristics appealing for industry applications. We therefore tried to achieve two sets of goals: 
  
1. **APPLICATIVE GOALS**

   1.1 Create a methdology able to **jointly** estimate engagement related behaviours (i.e. churn probability and survival time) given a          restricted number of observations and features.  
   1.2 Create a methodlogy which scale well with the number of data points, that is noise resilient and that can be easily integrated in a        larger framework.  
   1.3 Create a methodology able to explicitly model uncertainty around estimates.  
  
2. **THEORETICAL GOALS**  
   
   2.1 Evaluate the assumption that non-linear model are more suitable than linear ones.  
   2.2 Evaluate the assumption that features indicative of behavioural activity are good candidates for modelling churn probability and          survival time.  
   2.3 Evaluate the assumption that churn probability and survival time can be modelled as arising from a common underlying process.  
   2.4 Evaluate the assumption that the aformetioned process is better modelled as a dynamic system rather than a static one.  
  
# Proposed Model
Here a graphical reppresentation of the Deep Neural Network architecture we designed and developed for achieving the aformetioned goals  
  
<p align="center">   
  <img width="300" height="330" src="https://raw.githubusercontent.com/vb690/churn_survival_joint_estimation/master/figures/bm_architecture.jpg">
</p>  
  
The first section aims to learn an embedding for each game context and fuses it, via concatenation, with a restricted set of features indicative of behavioural activity. The embedding allows the model to learn a rich multi-dimensional representation of the game context projecting similar games into closer points in the latent space. The second section takes these fused representation over time and models them temporally using a Recurrent Neural Network (RNN) employing Long Short-Term Memory (LSTM) cells. The use of a RNN is particularly suitable here because it can handle time series of different lengths and explicitly model temporal dependencies. We thought to use this part of the model for extracting a high level representation of the player state which could be used for predicting measures of future disengagement and sustained engagement. This was achieved by ’branching’ two shallow Neural Networks tasked to perform churn probability and survival time estimation.  

# Data 
Due to commerical sensitivity and  [privacy policies](https://en.wikipedia.org/wiki/General_Data_Protection_Regulation) we are not allowed to freely share the data employed in our work.  
  
However, due to the principles that guided the features selection, we believe that their extreme generalizability makes relatively easy to test our methodology on different data sources.  
    
For running `minimal_test.py` the follwoing structure in the `data` folder should be respected:
```
n=Number of data points
k=Total number of unique contexts taken into consideration
t=Maximum number of time-steps taken into consideration

data   
│
└───collapsed
|
│   X_tr.npy        |   shape=(n, 5+k)  |   Training set features + one-hot encode of context
│   X_ts.npy        |   shape=(n, 5+k)  |   Test set features + one-hot encode of context
│   y_r_tr.npy      |   shape=(n, 1)    |   Training set regression target
│   y_r_ts.npy      |   shape=(n, 1)    |   Test set regression target
│   y_c_tr.npy      |   shape=(n, 1)    |   Test set regression target
│   y_c_ts.npy      |   shape=(n, 1)    |   Test set regression target
|   context_tr.npy  |   shape=(n, 1)    |   Training set context names
|   context_ts.npy  |   shape=(n, 1)    |   Test set context names
|
└───unfolded
|
│   X_tr.npy        |   shape=(n, (5*t)+k)  |   Training set features + one-hot encode of context
│   X_ts.npy        |   shape=(n, (5*t)+k)  |   Test set features + one-hot encode of context
│   y_r_tr.npy      |   shape=(n, 1)        |   Training set regression target
│   y_r_ts.npy      |   shape=(n, 1)        |   Test set regression target
│   y_c_tr.npy      |   shape=(n, 1)        |   Test set regression target
│   y_c_ts.npy      |   shape=(n, 1)        |   Test set regression target
|   context_tr.npy  |   shape=(n, 1)        |   Training set context names
|   context_ts.npy  |   shape=(n, 1)        |   Test set context names
|
└───temporal  
|
│   X_feat_tr.npy   |   shape=(n, t, 5)   |   Training set features (temporal format)
│   X_feat_ts.npy   |   shape=(n, t, 5)   |   Training set features (temporal format)
│   X_cont_tr.npy   |   shape=(n, 1)      |   Training set context features (numerical encoding)
│   X_cont_ts.npy   |   shape=(n, 1)      |   Training set context features (numerical encoding)
│   y_r_tr.npy      |   shape=(n, 1)      |   Training set regression target
│   y_r_ts.npy      |   shape=(n, 1)      |   Test set regression target
│   y_c_tr.npy      |   shape=(n, 1)      |   Test set regression target
│   y_c_ts.npy      |   shape=(n, 1)      |   Test set regression target
|   context_tr.npy  |   shape=(n, 1)      |   Training set context names
|   context_ts.npy  |   shape=(n, 1)      |   Test set context names
```
# Results
  
1. **PERFORMANCE**  
Up Next

2. **INSPECTING tTHE LEARNED USER EMBEDDING** 
One of the advantege of modelling engagement related behaviours as arising from a common underlying process is that we can interpret this last one as a reppresentation of the user state.  
  
<p align="center">  
    <img width="280" height="280" src="https://raw.githubusercontent.com/vb690/churn_survival_joint_estimation/master/figures/context_emb.gif" />
  <img width="280" height="280" src="https://raw.githubusercontent.com/vb690/churn_survival_joint_estimation/master/figures/churn_emb.gif" />
    <img width="280" height="280" src="https://raw.githubusercontent.com/vb690/churn_survival_joint_estimation/master/figures/survival_emb.gif" />
</p>

# Usage
Up Next
# Requirements
```
# Pipenv is a virtual environment manager
pip install pipenv

# Create a virtual environment in this directory
pipenv install

# open / activate virtual environment
pipenv shell

# install all the dependencies
pip install -r requirements.txt
# Now we are good to go....
```  
  
For Windows users we strongly advise to install numpy==1.17.1+mkl and scipy==1.3.1 (in this order) directly from the binaries distributed through https://www.lfd.uci.edu/~gohlke/pythonlibs.
# Cite & Contact
Please cite this work as:  
`Bonometti, Valerio, Ringer, Charles, Hall, Mark, Wade, Alex R., Drachen, Anders (2019) Modelling Early User-Game Interactions for Joint Estimation of Survival Time and Churn Probability, In: Proceedings of the IEEE Conference on Games 2019. IEEE`  
  
Or get in contact with us, we are looking for collaboration opportunities.
