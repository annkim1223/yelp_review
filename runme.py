import kfold_template
import pandas as pd
from sklearn import linear_model
import numpy as np
from sklearn import tree



from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals.six import StringIO
# from IPython.display import Image
# import pydotplus
from sklearn.tree import export_graphviz


import matplotlib
import matplotlib.pyplot as plt

# # just look at one business category
# select_df = business_df[business_df['Restaurants'] == 1]


################################# Import the data
df = pd.read_csv("cleaned_data_ydummies.csv")
# restauratnts_data= df[df[' Restaurants'] == 1]
# df = pd.DataFrame(restauratns_data) 

# print(list(df))
# print(df['stars'])


target = df.iloc[:,14].values # 
print("### target is:")
print(target)

target_cat = df.iloc[:,23].values


target_dummies = df.iloc[:,15:22].values # 
# print("### target_dummies is:")
# print(target_dummies)

data = df.iloc[:,1:13].values
print("### Data is:")
# print(list(data))
print(data.shape) 

# for i in data:
# 	# pd.to_numeric(data[i])
# 	data[i] = data[i].astype('int64')
# 	# dataTypeSeries = data.dtypes
# 	# print('Data type of each column of Dataframe :')
# 	# print(dataTypeSeries)



################################# <linear Regression>
machine_lr = linear_model.LinearRegression()
machine_lr.fit(data,target)
print("coefficients of linear_model is: ")
print(machine_lr.coef_)


"""
fit the data and target into the linear model. Then the machine is updated.
you have machine to learn the model using the data.
sklearn makes sure you fit the data with different models in the previous line. 
If you feed data AND target it will try to fit the data into target.
And after the machines is fed, if you put ONLY data, 
then the machines will use the model and guess. 

"""


no_stars_data = pd.read_csv("cleaned_no_stars.csv")
# restauratnts_no_stars= no_stars_data[no_stars_data[' Restaurants'] == 1]
# no_stars_data = pd.DataFrame(restauratnts_no_stars) 
X = no_stars_data.iloc[:,1:13].values
# from 14 to 4
print("### X is:")
print(list(X))
print(X.shape) 
# now we try to feed the model with new data.
# the format should be the same as the dimension of X. 



results = machine_lr.predict(X)
print("prediction of linear_model is: ")
print(results)


# print('RMSE:')
# print(np.sqrt(metrics.mean_squared_error(rg_y_test, rg.predict(rg_X_test))))



r2_scores_lr = kfold_template.run_kfold(5, data, target, linear_model.LinearRegression(), 0, 0)


print("r2 of linear_model is: ")
print(r2_scores_lr)




# ################################# <Random Forest>

machine_rf = RandomForestClassifier(n_estimators=100,criterion="gini", max_depth=10)

r2_scores_rf, accuracy_rf, confusion_matrices_rf = kfold_template.run_kfold(5, data, target_cat, machine_rf, 1, 1)


print("r2_scores of random forest is: ")
print(r2_scores_rf)
print("accuracy_scores of random forest is: ")
print(accuracy_rf)
print("confusion matrices of random forest is: ")
for i in confusion_matrices_rf:
	print(i)

machine_rf.fit(data,target_cat)
print(sorted(zip(map(lambda x: round(x, 4), machine_rf.feature_importances_)), reverse=True))
rf_feature_import = machine_rf.feature_importances_
print("feature_importances_ is: ")
print(rf_feature_import)


# feature_importances = pd.DataFrame(machine_rf.feature_importances_,index = target_cat.columns, columns=['importance']).sort_values('importance', ascending=False)



results_rf = machine_rf.predict(X)
print("results from the random forest is: ")
print(results_rf)


# ################################# <Decision Tree>


# machine_tree = tree.DecisionTreeClassifier(criterion="gini", max_depth=10)
# #if you do not specify max depth, you are likely to face overfitting. 
# machine_tree.fit(data,target_cat)
# # print(machine_tree.coef_)

# results = machine_tree.predict(X)
# # print("prediction of decision tree is: ")
# # print(results)


# # # r2_scores_tree, accuracy_scores_tree, confusion_matrices_tree= kfold_template.run_kfold(5, data, target, machine_tree, 1, 1)
# # r2_scores_tree, accuracy_scores_tree = kfold_template.run_kfold(3, data, target_cat, machine_tree, 1, 0)


# # print("r2_scores_tree is: ")
# # print(r2_scores_tree)
# # print("accuracy_scores_tree is: ")
# # print(accuracy_scores_tree)
# # # print("confusion_matrices_tree is: ")
# # # for confusion_matrix in confusion_matrices_tree:
# # # 		print(confusion_matrix)




