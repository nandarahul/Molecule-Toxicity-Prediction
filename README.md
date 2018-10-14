# Molecule Toxicity Prediction
### Problem Type:
Given a molecule with various features, need to predict if the molecule is toxic or non-toxic.  It is a classification problem with two classes - toxic (1) and non-toxic (0).

### Given Features:
We are provided the following features of each molecule:
- Maximum Degree
- Minimum Degree
- Molecular Weight
- Number of H-Bond Donors
- Number of Rings
- Number of Rotatables
- Polar Surface Area
- Inchi_key
- Graphs
- SMILES

### Feature Engineering:
In addition to the above features, using the rdkit package, I added the following features for each molecule that would further help in toxicity prediction [1] :

1) **fr_phos** (Number of phosphoric acid groups + phosphoric ester groups in the molecule)
This is basically (Fragments.fr_phos_acid(mol) + Fragments.fr_phos_ester(mol) ) where    ‘mol’ is an rdkit molecule object and ‘Fragments’ is a rdkit module under ‘Chem’ package.

2) **aromatic_carbocycles** (Number of aromatic carbocycles in the molecule)
	Computed using Lipinski.NumAromaticCarbocycles(mol) where ‘mol’ is an rdkit molecule object and ‘Lipinski’ is a rdkit module under ‘Chem’ package.

3) **MolLogP**
This refers to the Wildman-Crippen LogP value of a molecule. Computed using Crippen.MolLogP(mol) where ‘mol’ is an rdkit molecule object and ‘Crippen’ is a rdkit module under ‘Chem’ package.

4) **PEOE_VSA1**
PEOE_VSA1 is another descriptor value of a molecule computed using the ‘MolSurf’ module under rdkit ‘Chem’ package.

5) **Molecule Fingerprint**

### Background on Fingerprints:
Molecular fingerprints are a way of encoding the structure of a molecule. It is a series of binary digits (bits) that represent the presence or absence of particular substructures in the molecule. Each fingerprint bit corresponds to a fragment of the molecule.
The presence of some specific types of substructures in a molecule could determine the toxicity of a molecule. Hence, the fingerprint does that job of encoding the graph structure of a molecule as a vector of bits where each bit could be individually passed as a feature to our classifier.

### Extended-Connectivity Fingerprints (ECFP)
Extended-Connectivity Fingerprints (ECFPs) are circular topological fingerprints designed for molecular characterization, similarity searching, and structure-activity modeling.
After doing sufficient research, ECFP class of fingerprints seemed to be the best choice for toxicity prediction task.
The ‘AllChem’ module under the ‘Chem’ package of rdkit, provides the functions for generating the different types of fingerprint of a molecule.
Morgan fingerprint  - a type of circular fingerprint that is available as AllChem.GetMorganFingerprintAsBitVect() function, has been used in the project to generate a 2048 bit long string for a molecule. Each bit is considered as a separate feature.

### Final Features
A script (transform_data.py - included in the repo) is used to read the original training and testing CSV files and create new training and testing CSV files that have the new features introduced in the previous section for each molecule. It also eliminates the features that aren’t required/useful anymore such as Inchi_Key, Graph.
Finally the following features are used for training and testing:

Maximum Degree, Minimum Degree, Molecular Weight, Number of H-Bond Donors, Number of Rings, Number of Rotatables, Polar Surface Area, fr_phos, aromatic_carbocycles, MolLogP, PEOE_VSA1, Fingerprint (2048 length)

### Class Imbalance in Training Data:
An important observation is that there is a class imbalance in the training data. There are 7160 samples of negative class (non-toxic) and only 304 samples of positive class (toxic).
To avoid incorrect / skewed results, we assign unequal weights to the samples of both the classes.
Class_weight = ‘balanced’ is used or an appropriate dictionary of weights is constructed and passed to the loss function while training.

### Experiments
The listed experiments were conducted for testing the performance of different models for the specified task. From various classification algorithms available, I tried the following:
1) Logistic Regression (Using sklearn.linear_model.LogisticRegression)
2) Boosted Decision Tree (Using XGBClassifier from xgboost library)
3) Neural Network (Using PyTorch)
4) Random Forests (Using sklearn.ensemble.RandomForestClassifier)

### Hyper-parameter Tuning
For tuning the hyper-parameters of the 4 classifiers, I followed the **discrete-optimisation-within-validation** approach.  The procedure works as follows:
Using K-fold cross validation, iterate through the data. K was chosen as 7.
For each iteration of the 7-fold validation, call the 6-folds used for training the full training data. Split full training into training’ (80%) and validation’ (20%). Now, with training’ and validation’, find the best hyper-parameters for each of the 4 classifiers.
Compute the AUC score of each of the classifiers with the tuned parameters on the Validation data set (1-fold).
At the end of the 7 iterations, we have a list of AUC score on the Validation fold and the corresponding best parameters obtained for each classifier for every iteration.

We then select the best hyper-parameter using majority voting and to break any ties, we use the AUC score obtained for that hyper-parameter.

 **Here is the result obtained from every iteration of the discrete-optimisation-within-validation (using 7 folds):**
(For each classifier,  we get the AUC score on the Validation fold and the best hyper-parameters selected for that fold.)

**Fold: 0**
Logistic Regression: (0.83946503154714291, {'C': 0.01})
Gradient Boosted Decision Tree: (0.8682573535945971, {'learning_rate': 0.05, 'max_depth': 2})
Random Forest: (0.86916822180751807, {'max_depth': 7})
**Fold: 1**
Logistic Regression: (0.82032569092686392, {'C': 0.05})
Gradient Boosted Decision Tree: (0.85275037767706385, {'learning_rate': 0.05, 'max_depth': 4})
Random Forest: (0.87317826357415806, {'max_depth': 10})
**Fold: 2**
Logistic Regression: (0.82984537456678209, {'C': 0.01})
Gradient Boosted Decision Tree: (0.86551364080689597, {'learning_rate': 0.05, 'max_depth': 3})
Random Forest: (0.88656358304452154, {'max_depth': 9})
**Fold: 3**
Logistic Regression: (0.90049785173566121, {'C': 0.05})
Gradient Boosted Decision Tree: (0.91296687808315713, {'learning_rate': 0.1, 'max_depth': 3})
Random Forest: (0.90611289185932842, {'max_depth': 10})
**Fold: 4**
Logistic Regression: (0.83209438723317186, {'C': 0.05})
Gradient Boosted Decision Tree: (0.85058764691172795, {'learning_rate': 0.15, 'max_depth': 2})
Random Forest: (0.83448134760962966, {'max_depth': 10})
**Fold: 5**
Logistic Regression: (0.77311373297869923, {'C': 0.01})
Gradient Boosted Decision Tree: (0.74938052694991941, {'learning_rate': 0.05, 'max_depth': 2})
Random Forest: (0.76681670417604397, {'max_depth': 9})
**Fold: 6**
Logistic Regression: (0.82754972011104533, {'C': 0.01})
Gradient Boosted Decision Tree: (0.85738178673826959, {'learning_rate': 0.2, 'max_depth': 2})
Random Forest: (0.8691234697128295, {'max_depth': 8})

### Result:

| Classifier                     | Tuned Hyper-parameters              |
|--------------------------------|-------------------------------------|
| Logistic Regression            | C = 0.01                            |
| Gradient Boosted Decision Tree | Learning_rate = 0.05, max_depth = 2 |
| Neural Network                 | Learning_rate = 0.01, epochs = 10   |
| Random Forest                  | Max_depth = 9                       |

### Determining the Best Model (Multiple Hypothesis Test)
Once we have done the hyper-parameter tuning for all the 4 classifiers, we need to determine the best performing classifier that would be used to run on the final test data. This would involve a multiple hypothesis test. Here is the followed procedure:
Using Stratified K-fold Cross Validation, train the 4 classifiers on K-1 folds and test on Kth fold.
We take K as 10. In every iteration of the Cross Validation, we compute the AUC score of each classifier on the Validation set.
At the end, we have a list of AUC scores obtained for each of the 4 classifiers.
Now, we compute the mean AUC score.

Let 	M1 = mean AUC of Random Forest on the complete population
 	M2 = mean AUC of Gradient Boosted Decision Tree on the complete population
M3 =  mean AUC of Neural Network on the complete population
 	M4 =  mean AUC of Logistic Regression on the complete population

We take 𝛂 = 0.05/3 = 0.0167 (After applying Bonferroni’s correction)
The following tests were performed and the value of t-statistic and p value were computed using the stats module in scipy package.
From the observations, Random Forest seemed to be the best performing classifier and that’s why I perform the t-test of Random Forest vs the rest, one by one.

**Random Forest vs Gradient Boosted Decision Tree**
H0: M1 - M2 = 0
H1: M1 > M2
p = 0.48096968778656657
Since p value is more than 𝛂, we cannot reject the Null Hypothesis.

**Random Forest vs Neural Network**
H0: M1 - M3 = 0
H1: M1 > M3
p = 0.04688596131138213
Since p value is more than 𝛂, we cannot reject the Null Hypothesis.

**Random Forest vs Logistic Regression**
H0: M1 - M4 = 0
H1: M1 > M4
p = 0.00895185841193689
Since p value is more than 𝛂, we reject the Null Hypothesis in favour of the Alternative.

**From the multiple hypothesis test, we cannot really say which model is the best model. Random Forest and Gradient Boosted Decision trees are comparable in terms of performance.**
Ultimately, I selected Random Forest to run on the final test data.

## Plots
### Logistic Regression
<img src="/CurvePlots/LR_ROC.png" width="45%" > <img src="/CurvePlots/LR_learning_curves.png" width="45%" >


### Neural Network
<img src="/CurvePlots/NN_ROC.png" width="45%" /> <img src="/CurvePlots/NN_learning_curve.png" width="45%" />


### Random Forest
<img src="/CurvePlots/RF_ROC.png" width="45%" /> <img src="/CurvePlots/RF_learning_curve.png" width="45%" />


### Boosted Decision Tree
<img src="/CurvePlots/xgb_ROC.png" width="45%" /> <img src="/CurvePlots/XGB_learning_curves.png" width="45%" />


