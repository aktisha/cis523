import pandas as pd
import numpy as np
from sklearn.metrics import f1_score#, balanced_accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

class MappingTransformer(BaseEstimator, TransformerMixin):
  
  def __init__(self, mapping_column, mapping_dict:dict):
    assert isinstance(mapping_dict, dict), f'{self.__class__.__name__} constructor expected dictionary but got {type(mapping_dict)} instead.'
    self.mapping_dict = mapping_dict
    self.mapping_column = mapping_column  #column to focus on

  def fit(self, X, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return X

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
    assert self.mapping_column in X.columns.to_list(), f'{self.__class__.__name__}.transform unknown column "{self.mapping_column}"'  #column legit?
    
    #now check to see if all keys are contained in column
    column_set = set(X[self.mapping_column])
    keys_not_found = set(self.mapping_dict.keys()) - column_set
    if keys_not_found:
      print(f"\nWarning: {self.__class__.__name__}[{self.mapping_column}] does not contain these keys as values {keys_not_found}\n")

    #now check to see if some keys are absent
    keys_absent = column_set -  set(self.mapping_dict.keys())
    if keys_absent:
      print(f"\nWarning: {self.__class__.__name__}[{self.mapping_column}] does not contain keys for these values {keys_absent}\n")

    X_ = X.copy()
    X_[self.mapping_column].replace(self.mapping_dict, inplace=True)
    return X_

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result

class OHETransformer(BaseEstimator, TransformerMixin):
  def __init__(self, target_column, dummy_na=False, drop_first=False):
    self.target_column = target_column
    self.dummy_na = dummy_na
    self.drop_first = drop_first
  
  def fit(self, X, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return X

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
    assert self.target_column in X.columns.to_list(), f'{self.__class__.__name__}.transform unknown column "{self.target_column}"'
    
    X_ = pd.get_dummies(X,
                               prefix=self.target_column,
                               prefix_sep='_',
                               columns=[self.target_column],
                               dummy_na=self.dummy_na,
                               drop_first=self.drop_first
                               )
    
    return X_

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result
  
class DropColumnsTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, column_list, action='drop'):
    assert action in ['keep', 'drop'], f'{self.__class__.__name__} action {action} not in ["keep", "drop"]'
    self.column_list = column_list
    self.action = action

  def fit(self, X, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return X

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'

    columns_not_found = set(self.column_list) - set(X.columns.to_list())
    X_ = X.copy()

    
    if self.action == 'keep':
      if(set(self.column_list).issubset(set(X.columns.to_list()))) == False:
        print("lala")
        # change
        assert set(self.column_list) <= set(X.columns.to_list()), f'{self.__class__.__name__}.the dataframe does not contain "{columns_not_found}" to keep.'
      X_ = X_[self.column_list]
    elif self.action == 'drop':
      if columns_not_found:
        print(f"\nWarning: {self.__class__.__name__} does not contain {self.columns_not_found} to drop.\n")
      X_ = X.drop(columns = self.column_list, errors = 'ignore')
    
    return X_

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result
  
class PearsonTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, threshhold):
    self.threshhold = threshhold

  def fit(self, X, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return X

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
    
    df_corr = transformed_df.corr(method='pearson')
    masked_df = df_corr.abs() >= self.threshhold

    upper_mask = np.triu(masked_df, 1)

    correlated_columns = [masked_df.columns.values[j] for i, j in enumerate(set(np.where(upper_mask)[1]))]
    new_df = transformed_df.drop(columns=correlated_columns)

    return new_df

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result
  
class Sigma3Transformer(BaseEstimator, TransformerMixin):
  def __init__(self, column):
    self.column = column  #column to focus on

  def fit(self, X, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return X

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
    assert self.column in X.columns.to_list(), f'{self.__class__.__name__}.transform unknown column "{self.column}"'  #column legit?
    
    assert all([isinstance(v, (int, float)) for v in X[self.column].to_list()])

    mean = X[column].mean()
    sigma = X[column].std()
    sigma3min = mean - 3 * sigma
    sigma3max = mean + 3 * sigma

    X_ = X.copy()
    X_[self.column] = X[self.column].clip(lower = sigma3min, upper = sigma3max)
    return X_

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result
  
class TukeyTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, column, fence):
    self.column = column  #column to focus on
    self.fence = fence #fence to focus on

  def fit(self, X, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return X

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
    assert self.column in X.columns.to_list(), f'{self.__class__.__name__}.transform unknown column "{self.column}"'  #column legit?
    assert all([isinstance(v, (int, float)) for v in X[self.column].to_list()])

    q1 = X[self.column].quantile(0.25)
    q3 = X[self.column].quantile(0.75)

    iqr = q3-q1

    if self.fence == 'outer':
      low = q1-3*iqr
      high = q3+3*iqr
    elif self.fence == 'inner':
      low = q1-1.5*iqr
      high = q3+1.5*iqr

    X_ = X.copy()
    X_[self.column] = X[self.column].clip(lower = low, upper = high)
    return X_

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result

class MinMaxTransformer(BaseEstimator, TransformerMixin):
  from sklearn.preprocessing import MinMaxScaler

  def __init__(self):
    pass  #takes no arguments

  #fill in rest below

  def fit(self, X, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return X

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'

    X_copy = X.copy()
    X_column_names = X_copy.columns.tolist()
    scaler = self.MinMaxScaler()
    res = scaler.fit_transform(X_copy)
    X_copy = pd.DataFrame(res)
    X_copy.columns = X_column_names

    X_copy.describe(include='all').T

    return X_copy

  def fit_transform(self, X, y = None):
      result = self.transform(X)
      return result
    
    
class KNNTransformer(BaseEstimator, TransformerMixin):

  from sklearn.impute import KNNImputer

  def __init__(self, n_neighbors = 5, weights = "uniform"):
    #your code
    self.n_neighbors = n_neighbors
    self.weights = weights
  
  def fit(self, X, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return X

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
    
    X_copy = X.copy()
    X_column_names = X_copy.columns.tolist()
    imputer = self.KNNImputer(n_neighbors=self.n_neighbors, weights=self.weights, add_indicator=False)
    res = imputer.fit_transform(X_copy)
    X_copy = pd.DataFrame(res)
    X_copy.columns = X_column_names

    return X_copy

  def fit_transform(self, X, y = None):
      result = self.transform(X)
      return result
    
def find_random_state(features_df, labels, n=200):
  assert isinstance(features_df, pd.core.frame.DataFrame), f'expected a Dataframe but got {type(features_df)} instead.'

  var = []  #collect test_error/train_error where error based on F1 score

  for i in range(1, n):
      train_X, test_X, train_y, test_y = train_test_split(features_df, labels, test_size=0.2, shuffle=True,
                                                      random_state=i, stratify=labels)
      model.fit(train_X, train_y)  #train model
      train_pred = model.predict(train_X)           #predict against training set
      test_pred = model.predict(test_X)             #predict against test set
      train_f1 = f1_score(train_y, train_pred)   #F1 on training predictions
      test_f1 = f1_score(test_y, test_pred)      #F1 on test predictions
      f1_ratio = test_f1/train_f1          #take the ratio
      var.append(f1_ratio)

  rs_value = sum(var)/len(var)  #get average ratio value

  idx = np.array(abs(var - rs_value)).argmin()  #find the index of the smallest value
  return idx

#General dataset setup
def dataset_setup(full_table, label_column_name:str, the_transformer, rs, ts=.2):
  #your code below
  from sklearn.model_selection import train_test_split

  table_features = full_table.drop(columns=label_column_name)
  labels = full_table[label_column_name].to_list()

  X_train, X_test, y_train, y_test = train_test_split(table_features, labels, test_size=0.2, shuffle=True,
                                                    random_state=rs, stratify=labels)
  
  X_train_transformed = the_transformer.fit_transform(X_train)
  X_test_transformed = the_transformer.fit_transform(X_test)

  x_trained_numpy = X_train_transformed.to_numpy()
  x_test_numpy = X_test_transformed.to_numpy()
  y_train_numpy = np.array(y_train)
  y_test_numpy = np.array(y_test)

  return x_trained_numpy, x_test_numpy, y_train_numpy, y_test_numpy

#Titanic transformer
titanic_transformer = Pipeline(steps=[
    ('drop', DropColumnsTransformer(['Age', 'Gender', 'Class', 'Joined', 'Married',  'Fare'], 'keep')),
    ('gender', MappingTransformer('Gender', {'Male': 0, 'Female': 1})),
    ('class', MappingTransformer('Class', {'Crew': 0, 'C3': 1, 'C2': 2, 'C1': 3})),
    ('ohe', OHETransformer(target_column='Joined')),
    ('age', TukeyTransformer(column='Age', fence='outer')), #from chapter 4
    ('fare', TukeyTransformer(column='Fare', fence='outer')), #from chapter 4
    ('minmax', MinMaxTransformer()),  #from chapter 5
    ('imputer', KNNTransformer())  #from chapter 6
    ], verbose=True)

#Titanic dataset setup
def titanic_setup(titanic_table, transformer=titanic_transformer, rs=40, ts=.2):
  x_trained_numpy, x_test_numpy, y_train_numpy,  y_test_numpy = dataset_setup(titanic_table, 'Survived',
                                                                           titanic_transformer,
                                                                           rs,
                                                                           ts)
  return x_trained_numpy, x_test_numpy, y_train_numpy,  y_test_numpy

#Customer transformer
customer_transformer = Pipeline(steps=[
    ('id', DropColumnsTransformer(column_list=['ID'])),  #you may need to add an action if you have no default
    ('os', OHETransformer(target_column='OS')),
    ('isp', OHETransformer(target_column='ISP')),
    ('level', MappingTransformer('Experience Level', {'low': 0, 'medium': 1, 'high':2})),
    ('gender', MappingTransformer('Gender', {'Male': 0, 'Female': 1})),
    ('time spent', TukeyTransformer('Time Spent', 'inner')),
    ('minmax', MinMaxTransformer()),
    ('imputer', KNNTransformer())
    ], verbose=True)

#Customer dataset setup
def customer_setup(customer_table, transformer=customer_transformer, rs=76, ts=.2):
  x_trained_numpy, x_test_numpy, y_train_numpy,  y_test_numpy = dataset_setup(customer_table, 'Rating',
                                                                           customer_transformer,
                                                                           rs,
                                                                           ts)
  return x_trained_numpy, x_test_numpy, y_train_numpy,  y_test_numpy

