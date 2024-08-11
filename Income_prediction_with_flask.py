import pandas
import numpy
from sklearn import preprocessing
 
df = pandas.read_csv('adult.csv')
df.head()


df = df.drop(['fnlwgt', 'educational-num','capital-gain','capital-loss'], axis=1)
 
col_names = df.columns
 
for c in col_names:
    df = df.replace("?", numpy.NaN)
df = df.apply(lambda x: x.fillna(x.value_counts().index[0]))


df.replace(['Divorced', 'Married-AF-spouse',
            'Married-civ-spouse', 'Married-spouse-absent',
            'Never-married', 'Separated', 'Widowed'],
           ['divorced', 'married', 'married', 'married',
            'not married', 'not married', 'not married'], inplace=True)

category_col = ['workclass', 'race', 'education', 'marital-status', 'occupation',
                'relationship', 'gender', 'native-country', 'income']
labelEncoder = preprocessing.LabelEncoder()

mapping_dict = {}
for col in category_col:
    # .fit_transorm allocates numerical values to the unique categorical values
    # transorm just converts all the categorical data into numerical values
    # without transorm if you apply fit then it raises error
    df[col] = labelEncoder.fit_transform(df[col])
 
    le_name_mapping = dict(zip(labelEncoder.classes_,# it just contains the classes that are present the unique classes but not the data to transform so we don't directly apply .fit_transform on labelEncoder.classes_
                               labelEncoder.transform(labelEncoder.classes_)))
 
    mapping_dict[col] = le_name_mapping
print(mapping_dict)


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
 
X = df.values[:, 0:12]
Y = df.values[:, 10]



X_train, X_test, y_train, y_test = train_test_split(
           X, Y, test_size = 0.3, random_state = 100)
 
dt_clf_gini = DecisionTreeClassifier(criterion = "gini",
                                     random_state = 100,
                                     max_depth = 5,
                                     min_samples_leaf = 5)
 
dt_clf_gini.fit(X_train, y_train)
y_pred_gini = dt_clf_gini.predict(X_test)
 
print ("Decision Tree using Gini Index\nAccuracy is ",
             accuracy_score(y_test, y_pred_gini)*100 )

#creating and training a model
#serializing our model to a file called model.pkl
import pickle
pickle.dump(dt_clf_gini, open("model.pkl","wb"))