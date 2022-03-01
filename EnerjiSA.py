import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, r2_score, roc_auc_score, roc_curve, \
    classification_report
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale, StandardScaler, MinMaxScaler, RobustScaler
from sklearn import model_selection
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, GradientBoostingClassifier, \
    RandomForestClassifier
from sklearn import neighbors
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import lightgbm as lgb
from xgboost import XGBRegressor
from xgboost import plot_importance
import statsmodels.api as sm
import statsmodels.tsa.api as smt
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings

warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.8f' % x)


def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


gen = pd.read_csv('../input/enerjisa-enerji-veri-maratonu/generation.csv', delimiter=';', decimal=',')
temp = pd.read_csv('../input/enerjisa-enerji-veri-maratonu/temperature.csv', delimiter=';', decimal=',')
df_train = pd.concat([gen.iloc[:25560], temp.iloc[:25560].drop('DateTime', axis=1)], axis=1)
df_test = temp.iloc[25560:26304]


def preprocess(df):
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df = df.set_index('DateTime')
    df['WWCode'] = df['WWCode'].fillna(0).astype(int)
    return df


df_train = preprocess(df_train)
df_test = preprocess(df_test)

df_test.loc[df_test['WWCode'] == 84, 'WWCode'] = 83

check_df(df_train)

check_df(df_test)


def create_features(df, label=None):
    """
    Creates time series features from datetime index
    """
    df_copy = df.copy()
    df_copy['date'] = df_copy.index
    df_copy['hour'] = df_copy['date'].dt.hour
    df_copy['dayofweek'] = df_copy['date'].dt.dayofweek
    df_copy['quarter'] = df_copy['date'].dt.quarter
    df_copy['month'] = df_copy['date'].dt.month
    df_copy['year'] = df_copy['date'].dt.year
    df_copy['dayofyear'] = df_copy['date'].dt.dayofyear
    df_copy['dayofmonth'] = df_copy['date'].dt.day
    df_copy['weekofyear'] = df_copy['date'].dt.weekofyear

    X = df_copy[['hour', 'dayofweek', 'quarter', 'month', 'year',
                 'dayofyear', 'dayofmonth', 'weekofyear']]

    if label:
        y = df_copy[label]
        X = pd.concat([df.drop(label, axis=1), X], axis=1)
        return X, y
    else:
        X = pd.concat([df, X], axis=1)
        return X


def split_train(df, split_date):
    train = df.loc[df.index <= split_date].copy()
    val = df.loc[df.index > split_date].copy()
    return train, val


split_date = '2021-05-01'
df_train, df_val = split_train(df_train, split_date)

x_train, y_train = create_features(df_train, label='Generation')
x_val, y_val = create_features(df_val, label='Generation')
x_test = create_features(df_test)

lgb_params = {'metric': {'rmse'},
              'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'num_boost_round': 20000,
              'early_stopping_rounds': 200,
              'nthread': -1}

datasets = {'x_train': x_train,
            'y_train': y_train,
            'x_val': x_val,
            'y_val': y_val,
            'x_test': x_test
            }

y_train.shape, x_train.shape, y_val.shape, x_val.shape

lgbtrain = lgb.Dataset(data=x_train, label=y_train)

lgbval = lgb.Dataset(data=x_val, label=y_val, reference=lgbtrain)

model = lgb.train(lgb_params, lgbtrain,
                  valid_sets=[lgbtrain, lgbval],
                  num_boost_round=lgb_params['num_boost_round'],
                  early_stopping_rounds=lgb_params['early_stopping_rounds'],
                  verbose_eval=100)

y_pred_val = model.predict(x_val, num_iteration=model.best_iteration)
y_pred_val


def plot_lgb_importances(model, plot=False, num=10):
    gain = model.feature_importance('gain')
    feat_imp = pd.DataFrame({'feature': model.feature_name(),
                             'split': model.feature_importance('split'),
                             'gain': 100 * gain / gain.sum()}).sort_values('gain', ascending=False)
    if plot:
        plt.figure(figsize=(10, 10))
        sns.set(font_scale=1)
        sns.barplot(x="gain", y="feature", data=feat_imp[0:25])
        plt.title('feature')
        plt.tight_layout()
        plt.show()
    else:
        print(feat_imp.head(num))


plot_lgb_importances(model, num=30)
plot_lgb_importances(model, num=30, plot=True)

lgb.plot_importance(model, max_num_features=20, figsize=(10, 10), importance_type="gain")
plt.show()

lgb_params = {'metric': {'mae'},
              'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'nthread': -1,
              "num_boost_round": model.best_iteration}

lgbtrain_all = lgb.Dataset(data=x_train, label=y_train)
model = lgb.train(lgb_params, lgbtrain_all, num_boost_round=model.best_iteration)

test_preds = model.predict(x_test, num_iteration=model.best_iteration)


def clip_nights(pred, test=False):
    nights = [21, 22, 23, 0, 1, 2, 3, 4]
    if not test:
        for time, row in pred.iterrows():
            if time.hour in nights or row['pred'] < 0:
                row['pred'] = 0
        return pred
    else:
        for i, hour in enumerate(pd.to_datetime(pred['DateTime'])):
            if hour.hour in nights or pred.loc[i, 'Generation'] < 0:
                pred.loc[i, 'Generation'] = 0
        return pred


pred_val = model.predict(datasets['x_val'])
pred_val = pd.DataFrame(pred_val, index=datasets['x_val'].index, columns=['pred'])

rmse = mean_squared_error(datasets['y_val'], pred_val, squared=False)
print('RMSE: ', rmse)
