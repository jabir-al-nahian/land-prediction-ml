import pandas as pd
import numpy as np
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('/content/Dataset.csv')

df.head(5)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(df.drop('Disease',axis=1))

scaled_features = scaler.transform(df.drop('Disease',axis=1))

df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])
df_feat.head()

plt.scatter(df_feat, scaled_features)
plt.show()

import seaborn as sns

sns.pairplot(df,hue='Disease')

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(scaled_features,df['Disease'],
                                                    test_size=0.30, random_state=0)

"""# **Random Forest**"""

from sklearn.ensemble import RandomForestClassifier
rfmodel = RandomForestClassifier()
rfmodel.fit(X_train,y_train)

rfmodel.score(X_test, y_test)

"""# **SVM**"""

from sklearn.svm import SVC
smodel = SVC()
smodel.fit(X_train,y_train)

smodel.score(X_test, y_test)

"""# **KNN**"""

from sklearn.neighbors import KNeighborsClassifier
knnmodel = KNeighborsClassifier(n_neighbors=33)
knnmodel.fit(X_train,y_train)

knnmodel.score(X_test, y_test)

"""# **Tree**"""

from sklearn.tree import DecisionTreeClassifier
tmodel = DecisionTreeClassifier()
tmodel.fit(X_train,y_train)

tmodel.score(X_test, y_test)
