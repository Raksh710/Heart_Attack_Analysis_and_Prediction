#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv('heart.csv')


# In[3]:


df.head(2)


# In[4]:


o2 = pd.read_csv('o2Saturation.csv')


# In[5]:


len(df)


# In[7]:


len(o2)


# In[8]:


df.info()


# In[9]:


df.isna().sum()


# In[11]:


o2.head(2)


# In[10]:


o2.describe()


# In[13]:


df['output'].unique()


# In[15]:


len(df[df['sex']==1])


# In[16]:


len(df[df['sex']==0])


# # I guess depending on the data 0 means Female and 1 means male. Because this dataset is from Bangladesh where number of males (in general) is much much greater than that of women. So, going by this trend I guess, this will hold true for heart patients as well.

# In[20]:


df.corr()['output'].sort_values()[:-1]


# In[21]:


df.describe()


# In[22]:


96/303


# # the dataset's target variable distribution is around 70%-30% (70% of all values being 1)

# # EDA

# In[23]:


sns.histplot(data=df,x='output',kde=True)


# In[26]:


sns.countplot(data=df , x = 'cp', hue ='output')


# # TEST TRAIN SPLIT

# In[27]:


from sklearn.model_selection import train_test_split


# In[29]:


X = df.drop('output',axis=1)
y = df['output']


# In[33]:


X.head()


# # creating dummies

# In[36]:


df['thall'].nunique()


# In[40]:


X_dummies = pd.get_dummies(data=X , columns=['sex','cp','fbs','restecg','exng','slp','caa','thall'] , drop_first=True)


# In[41]:


len(X_dummies.columns)


# In[42]:


X_dummies.head(2)


# In[ ]:





# In[43]:


X_train, X_test, y_train, y_test = train_test_split(X_dummies, y, test_size=0.3, random_state=42)


# In[44]:


from sklearn.preprocessing import StandardScaler


# In[45]:


sc = StandardScaler()


# In[46]:


X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# # LET'S APPLY CLASSIFICATION MODELS AND ASSESS THEIR ACCURCACY/ F1 METRIC

# # 1) LOGISTIC REGRESSION

# In[48]:


from sklearn.linear_model import LogisticRegression


# In[85]:


lr = LogisticRegression(fit_intercept=True,random_state=42)


# In[86]:


lr.fit(X_train,y_train)


# In[87]:


lr_pred = lr.predict(X_test)


# In[88]:


from sklearn.metrics import classification_report,confusion_matrix,f1_score


# In[89]:


print(classification_report(y_test,lr_pred))
print('\n')
print(confusion_matrix(y_test,lr_pred))
print('\n')
print(f1_score(y_test,lr_pred))


# In[91]:


from sklearn.model_selection import cross_val_score


# In[433]:


lr_accuracies = cross_val_score(lr , X_test , y_test , cv = 41)
print(lr_accuracies.mean())


# # 2) KNN

# In[90]:


from sklearn.neighbors import KNeighborsClassifier


# In[144]:


knn = KNeighborsClassifier(n_neighbors=30)


# In[145]:


knn.fit(X_train,y_train)


# In[146]:


knn_pred = knn.predict(X_test)


# In[147]:


print(confusion_matrix(y_test,knn_pred))
print('\n')
print(classification_report(y_test,knn_pred))


# In[441]:


Accuracy = []
for i in range(1,41):
    knn = KNeighborsClassifier(i)
    knn.fit(X_train,y_train)
    knn_pred = knn.predict(X_test)
    knn_accuracies = cross_val_score(knn , X_test , y_test , cv = 41)
    knn_accuracy_i = knn_accuracies.mean()
    Accuracy.append(knn_accuracy_i)


# In[456]:


knn_df = pd.DataFrame({'Neighbours': np.arange(1,41) , 'Accuracy': Accuracy})


# In[457]:


sns.scatterplot(data=knn_df , x = 'Neighbours',y='Accuracy')


# In[458]:


knn_df[knn_df['Accuracy']==knn_df['Accuracy'].max()]


# # implementing the best knn model with k=7

# In[446]:


knn = KNeighborsClassifier(14)
knn.fit(X_train,y_train)
knn_pred = knn.predict(X_test)
print(confusion_matrix(y_test,knn_pred))
print('\n')
print(classification_report(y_test,knn_pred))


# # CV Accuracy

# In[447]:


knn_accuracies = cross_val_score(knn , X_test , y_test , cv = 41)
print(knn_accuracies.mean())


# # KNN WITH K==14 IS BETTER THAN LOGISTIC REGRESSION WITH A CV ACCURACY OF 86.99%

#  # 3) SVC

# In[155]:


from sklearn.svm import SVC


# In[156]:


svc = SVC()


# In[157]:


svc.fit(X_train,y_train)


# In[158]:


svc_pred = svc.predict(X_test)


# In[159]:


print(confusion_matrix(y_test,svc_pred))
print('\n')
print(classification_report(y_test,svc_pred))


# # USING GRID SEARCH CV

# In[160]:


from sklearn.model_selection import GridSearchCV


# In[180]:


params = {'C': [10000,100000,1000000] , 'gamma': [0.000001 , (10**-7) , (10**-8)]}


# In[181]:


grid = GridSearchCV(SVC() , param_grid = params , verbose= 3 , cv=10)


# In[182]:


grid.fit(X_train,y_train)


# In[183]:


grid.best_params_


# In[184]:


grid.best_score_


# In[213]:


svc = SVC(C = 1e06 , gamma= 1e-07)


# In[214]:


svc.fit(X_train,y_train)


# In[215]:


svc_pred = svc.predict(X_test)


# In[216]:


print(confusion_matrix(y_test,svc_pred))
print('\n')
print(classification_report(y_test,svc_pred))


# In[431]:


svc_accuracies = cross_val_score(svc , X_test , y_test , cv = 41)
print(svc_accuracies.mean())


# # CV ON SVC PERFORMED WORST AMONG LOGISTIC REGRESSION AND KNN

# # 4) RANDOM FOREST

# In[227]:


from sklearn.ensemble import RandomForestClassifier


# In[320]:


rf = RandomForestClassifier(n_estimators=500,random_state=0 , max_depth=5)


# In[321]:


rf.fit(X_train,y_train)


# In[322]:


rf_pred = rf.predict(X_test)


# In[323]:


print(confusion_matrix(y_test,rf_pred))
print('\n')
print(classification_report(y_test,rf_pred))


# In[430]:


rf_accuracies = cross_val_score(rf , X_test , y_test , cv = 41)
print(rf_accuracies.mean())


# # RANDOM FOREST ALMOST LIKE LOGISTIC REGRESSION (LOGISTIC REGRESSION 4 MORE CORRECT VALUES THAN RF)..TILL NOW KNN SEEMS TO BE THE BEST FIT

# # 4) CATBOOST

# In[325]:


from catboost import CatBoostClassifier


# In[393]:


cb = CatBoostClassifier(iterations=1000 , random_state=0 , loss_function='Logloss' , depth=5)


# In[394]:


cb.fit(X_train,y_train , eval_set=(X_test,y_test) , plot=True)


# In[395]:


cb_pred = cb.predict(X_test)


# In[396]:


print(confusion_matrix(y_test,cb_pred))
print('\n')
print(classification_report(y_test,cb_pred))


# In[426]:


cb_accuracies = cross_val_score(cb , X_test , y_test , cv = 40)
print(cb_accuracies.mean())


# # SURPRISINGLY CATBOOST PERFORMED WORST OF ALL MODELS WITH ONLY 78.33% CV ACCURACY

# # HENCE, THE BEST MODEL IS KNN

# In[448]:


knn = KNeighborsClassifier(14)
knn.fit(X_train,y_train)
knn_pred = knn.predict(X_test)
print(confusion_matrix(y_test,knn_pred))
print('\n')
print(classification_report(y_test,knn_pred))


# In[455]:


knn_accuracies = cross_val_score(knn , X_test , y_test , cv = 41)
print(knn_accuracies.mean())


# # MEAN ACCURACY IS 86.99% OUT OF 41 CV FOLDS

# In[ ]:




