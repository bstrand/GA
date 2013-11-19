**GA Data Science Final Project Proposal**  
**Brant Strand, SF Section 2**

## Prediction of SaaS Subscription Renewal

### Summary 

I intend to develop a model to predict whether a given customer of a subscription software service will renew based upon account attributes and usage patterns. This prediction is intended to inform the business' understanding with an eye toward customer retention and identifying customer classes with preferable long term earning potential to inform first sales efforts.  

### Data Sets
A customer's profile will be composed by extracting features from the following data sets:-

Accounts
: A set of accounts whose subscription term ended in the prior 6 months

CRM
: Support cases logged, other products purchased, number of purchases, price paid, &c.

Account Attributes
: Features licensed, licensed capacity, pricing model, subscription length, month of purchase, &c.

Usage Metrics
: Number of meetings, subscription lifetime usage time, median usage time per day, peak concurrent users, median concurrent users, mean concurrent users

Environmental Factors
: Number of service outages on customer's cluster, number of downtime maintenances on customer's cluster

Application Interactions
: Keyword extraction from chat logs, 

### Approach

#### Data
1. Extract data from multiple sources (outlined above)
2. Transform as needed: cleaning, normalizing, and joining records
3. Load resulting customer representation data frames into data store (HDF5 files probably, maybe MongoDB or Redis)

#### Analysis

##### Summary Statistics & Visualization

Calculate and plot the feature data and their summary statistics to inform feature selection.

##### Methods
I intend to use supervised learning methods, initially evaluating SVM, Naive Bayes, and Random Forest classifiers. I will select the most promising of these and iterate toward improved accuracy through feature engineering and auxilliary techniques to refine the model. 

If time permits, this data may lend itself to an ensemble technique such as one in which one of several models is chosen by a shallow decision tree.

### Artifacts

#### Model
An accurate model with a reasonable practical and runtime for performing O(100) predictions.

#### Interface
Expose the model through a Python interface and package as a library.

#### Predictor
Provide a simple web interface or CLI with which a non-technical user can make predictions given the requisite inputs.

#### Report on Next Actions
A summary of the next actions for refinement or extension of the model and other avenues of future investigation.
