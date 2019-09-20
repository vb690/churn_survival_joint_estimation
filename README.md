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

One of the advantege of modelling engagement related behaviours as arising from a common underlying process is that we can interpret this last one as a reppresentation of the user state.  
  
<p align="center">  
    <img width="250" height="250" src="https://raw.githubusercontent.com/vb690/churn_survival_joint_estimation/master/figures/cont_emb.gif" />
  <img width="250" height="250" src="https://raw.githubusercontent.com/vb690/churn_survival_joint_estimation/master/figures/chu_emb.gif" />
    <img width="250" height="250" src="https://raw.githubusercontent.com/vb690/churn_survival_joint_estimation/master/figures/surv_emb.gif" />
</p>


# Data 
Up Next
# Usage
Up Next
# Requirements
Up Next
# Cite & Contact
Please cite this work as:  
`Bonometti, Valerio, Ringer, Charles, Hall, Mark, Wade, Alex R., Drachen, Anders (2019) Modelling Early User-Game Interactions for Joint Estimation of Survival Time and Churn Probability, In: Proceedings of the IEEE Conference on Games 2019. IEEE`  
  
Or get in contact with us, we are looking for collaboration opportunities.
