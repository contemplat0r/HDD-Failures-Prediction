#!/usr/bin/env python
# coding: utf-8

# #### Предсказание отказов жёстких дисков  с использованием S.M.A.R.T данных собраных в течении года в одном из датацентров [BACKBLAZE](https://www.backblaze.com/b2/hard-drive-test-data.html)

# Формально, наша задача - Panel Data Forecasting. Panel Data - это обобщение временных рядов (Time Series), когда независимых
# переменных несколько. Либо же наоборот - временные ряда, являються частным случаем Panel Data.
# Большая часть методов обращения с Panel Data предполагают заранее известным вид взаимосвязи между текущими значениями переменных
# и предшествующими а так же вид взаимосвязи между самими переменными (например - нет взаимосвязи, линейная взаимосвязь и т.д.).
# Здесь используется другой подход: мы не предполагаем заранее известным вид взаимосвязи, как раз ёе мы и должны найти или точнее
# аппроксимировать используя Machine Learning.

# Импортируем необходимые библиотеки

# In[45]:


from collections import Counter
import glob
import random
import os
import time
import gc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn

from sklearn.model_selection import GridSearchCV, learning_curve, validation_curve, KFold
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.externals import joblib
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, auc, roc_auc_score, f1_score, classification_report, confusion_matrix

import xgboost as xgb
from xgboost import XGBClassifier

import lightgbm
from lightgbm import LGBMClassifier


# Производим необходимые настройки, присваемаем значения глобальным переменным, создаём список файлов с данными.

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


sns.set()


# In[4]:


DATA_DIR = '../data/unarch/data_Q*_2019/'


# In[5]:


num_notfeature_columns = 4


# In[6]:


min_list_of_columns = [
    'date',
    'serial_number',
    'model',
    'capacity_bytes',
    'failure',
    'smart_3_normalized',
    'smart_187_normalized',
    'smart_193_normalized',
    'smart_195_normalized',
    'smart_240_normalized',
    'smart_241_normalized'
]


# In[119]:


datafiles = sorted(glob.glob(DATA_DIR + '*.csv'))


# In[ ]:


Определяем функции необходимые для дальнейшей работы.


# In[8]:


def compute_num_of_nans(datafiles):
    df = pd.read_csv(datafiles[0])
    total_len = len(df)
    num_of_nans = {col_name:len(df.loc[df[col_name].isna()]) for col_name in df.columns.to_list()}
    common_col_names_set = set(num_of_nans)
    all_col_names_list = df.columns.to_list()
    for datafile in datafiles[1:]:
        df = pd.read_csv(datafile)
        total_len += len(df)
        col_names = df.columns.to_list() #[num_n:]
        common_col_names_set = common_col_names_set.intersection(set(col_names))
        for col_name in col_names:
            if col_name in num_of_nans:
                num_of_nans[col_name] += len(df.loc[df[col_name].isna()])
            else:
                num_of_nans[col_name] = len(df.loc[df[col_name].isna()])
            if col_name not in all_col_names_list:
                col_names_idx = col_names.index(col_name)
                precend_col_name = col_names[col_names_idx - 1]
                all_col_names_idx = all_col_names_list.index(precend_col_name)
                all_col_names_list.insert(all_col_names_idx, col_name)
    common_col_names_list = [col_name for col_name in all_col_names_list if col_name in common_col_names_set]            
    num_of_nans = {col_name:num_of_nans[col_name] for col_name in common_col_names_set}
    gc.collect()
    return total_len, num_of_nans, common_col_names_list


# In[9]:


def drop_dominate_nans_columns(df, drop_column_nan_threshold_ratio=None):
    if drop_column_nan_threshold_ratio:
        drop_column_nan_threshold = df.shape[0] // drop_column_nan_threshold_ratio
    else:
        drop_column_nan_threshold = df.shape[0] // 2
    return df[
        [col for col in df.columns if len(df.loc[df[col].isna()]) < drop_column_nan_threshold]
    ]


# In[10]:


def fill_column_nans(df, column_name):
    #calculate number of NaNs in column
    na_len = len(df.loc[df[column_name].isna(), column_name])
    #create a pandas series containing values and corresponding quantities from not NaNs part of the column
    count_notna = df.loc[df[column_name].notna(), column_name].value_counts()
    #calculate relative frequencies (probabilities) of each value
    frequencies = count_notna / count_notna.sum()
    #make array that contain fill values with the same relative frequencies as in 'not NaN' column part
    fill_values = np.array(
        [np.random.choice(frequencies.index, p=frequencies.values) for _ in range(na_len)]
    )
    #fill NaNs
    df.loc[df[column_name].isna(), column_name] = fill_values
    return df


# In[11]:


def fill_nans(df, column_names):
    for column_name in column_names:
        fill_column_nans(df, column_name)
    return df


# In[12]:


def fill_nans(df):
    columns_with_nans = df.columns.to_list()[num_notfeature_columns:]
    fill_values = {col_name: np.round(df[col_name].astype(np.float64).mean()) for col_name in columns_with_nans}
    return df.fillna(value=fill_values)


# In[13]:


def reduce_mem_usage(df):
    """ 
    iterate through all the columns of a dataframe and 
    modify the data type to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    #print(('Memory usage of dataframe is {:.2f}MB').format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')
    end_mem = df.memory_usage().sum() / 1024**2
    #print(('Memory usage after optimization is: {:.2f} MB').format(end_mem))
    #print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


# In[14]:


def create_data_df(datafiles, total_len, num_of_nans, common_col_names_list, drop_column_nan_threshold_ratio=None):
    
    if drop_column_nan_threshold_ratio:
        drop_column_nan_threshold = total_len // drop_column_nan_threshold_ratio
    else:
        drop_column_nan_threshold = total_len // 2
    
    not_dominate_nan_columns = [
        col_name for col_name, col_num_of_nans in num_of_nans.items() if col_num_of_nans < drop_column_nan_threshold
    ]
    
    not_dominate_nan_columns = [col_name for col_name in common_col_names_list if col_name in not_dominate_nan_columns]
    
    data_df = reduce_mem_usage(pd.read_csv(datafiles[0], usecols=not_dominate_nan_columns))
    
    for datafile in datafiles[1:]:
        df = pd.read_csv(datafile, usecols=not_dominate_nan_columns)
        data_df = pd.concat([data_df, reduce_mem_usage(df)])
    not_dominate_nan_columns[3] = 'failure'
    not_dominate_nan_columns[4] = 'capacity_bytes'
    
    data_df = data_df[not_dominate_nan_columns]
    gc.collect()
    return data_df.reset_index().drop(['index'], axis=1)


# In[15]:


def scale_and_one_hot_encoding(df, scaler):
    features_columns = df.columns.to_list()[num_notfeature_columns:]
    
    scaled_features = pd.DataFrame(
        scaler.fit_transform(df[features_columns]), columns=features_columns, index=df.index, dtype=np.float16
    )
    
    return pd.concat(
        [df[df.columns.to_list()[:num_notfeature_columns]], scaled_features], axis=1
    ).reset_index().drop(['index', 'model', 'date', 'serial_number'], axis=1)


# In[16]:


def get_precedings_num(begin, end):
    return random.randint(begin, end)


# In[17]:


def select_failure_events_with_preceding(df, preceding_min_subset_len=10, preceding_max_subset_len=40, preceding_len_random=True):
    preceding_subset_len = preceding_min_subset_len
    if preceding_len_random:
        preceding_subset_len = random.randint(preceding_min_subset_len, preceding_max_subset_len)
    selected_df = pd.DataFrame()
    failure_events = df[df['failure'] == 1]
    failure_indexes = failure_events.index.to_list()
    for idx in failure_indexes:
        preceding = df.loc[(idx - preceding_subset_len):(idx - 1)]
        preceding_indexes = preceding.index.to_list()
        intersect = set(failure_indexes).intersection(set(preceding_indexes))
        if intersect:
            selected_df = pd.concat([selected_df, df.loc[(max(intersect) + 1):idx]])
        else:
            selected_df = pd.concat([selected_df, df.loc[(idx - preceding_subset_len):idx]])
    return selected_df.reset_index().drop(['index'], axis=1)


# In[18]:


def divide_features_and_target(df):
    return df[df.columns.to_list()[num_notfeature_columns:]], df['failure']


# In[19]:


def match_exactly(y_test, y_pred):
    false_negatives = 0
    false_positives = 0
    true_positives = 0
    true_negatives = 0
    for y_t, y_p in zip(y_test, y_pred):
        if y_t == 1 and y_p == 0:
            false_negatives += 1
        elif y_t == 0 and y_p == 1:
            false_positives += 1
        elif y_t == 1 and y_p == 1:
            true_positives += 1
        elif y_t == 0 and y_p == 0:
            true_negatives += 1
    print(
        "false_negatives: {}, false_positives: {}, true_negatives: {}, true_positives: {}".format(
            false_negatives, false_positives, true_negatives, true_positives)
    )


# Вычисляем количество NaNs в каждом столбце за весь год. Так же (на всякий случай) находим имена столбцов которые встречаються
# во всех файлах с данными. И вычисляем количество строк в общем датафрейме которые будет создан при обьединении данных содержашихся
# во всех файлах с данными.

# In[120]:


start_time = time.time()
data_df_total_len, num_of_nans, common_col_names_list = compute_num_of_nans(datafiles)
print("time_elapsed: {} min ".format((time.time() - start_time) / 60))


# Создаём объединённый датафрейм отбрасывая столбцы в которых NaN-ов больше некоторого порога - по умолчанию больше половины 
# количества строк в объединённом датафрейме. Так же производим оптимизацию памяти занимаемой объединённым датафреймом.

# In[121]:


start_time = time.time()
data_df = create_data_df(datafiles, data_df_total_len, num_of_nans, common_col_names_list)
print("time_elapsed: {} min ".format((time.time() - start_time) / 60))


# Сохраняем объединённый датафрейм, что бы в дальнейшем не создавать его каждый раз заново.

# In[122]:


data_df.to_hdf('data_df.hdf5', key='df', mode='w')


# Заполняем NaN-ы в неотброшенных столбцах - т.е. таких в которых NaN-ов меньше половины длинны столбца. Лучший подход здесь - 
# заполение NaN-ов значениями из not NaNs части столбца пропорционально частоте (вероятности) каждого значения. Но это очень медленно , к тому же большая часть столбцов содержит весьма значительное количество уникальных значений, что обессмысливает данный подход.
# Поэтому NaN-ы заполняються средним арифметическим not NaN значений в столбце.

# In[125]:


start_time = time.time()
ordered_not_nans_data_df = fill_nans(data_df)
print("time_elapsed: {} min ".format((time.time() - start_time) / 60))


# In[126]:


gc.collect()


# In[ ]:


standard_scaler = StandardScaler()
min_max_scaler = MinMaxScaler()


# Масштабирование данных должно улучшать предсказательную силу ML модели. Так же есть предположение, что частота отказов дисков может  существенно зависить от модели диска. Для того что бы ML модель учла модель диска необходимо использовать one hot encoding. Но строк в объединённом датафрейме 40737546 уникальных значений модели диска 55, при использовании one hot encoding в объединённый датафрейм добавится 55 столбца каждый длинной 40737546, что катастрофически увеличит объём памяти занимаемой датафреймом, было принято решение просто не учитывать модель диска при построении ML модели. В дальнейшем имело бы смысл произвести исследование влияния модел диска на аварийность "вручную" используя аппарат математической статистики.
# Масштабирование в этой версии решения задачи так же не производится, для того что бы посмотреть на предсказательную силу ML модели в наихудшем случае.

# In[ ]:


start_time = time.time()
data_df = scale_and_one_hot_encoding(data_df, min_max_scaler)
print("time_elapsed: {} min ".format((time.time() - start_time) / 60))


# In[ ]:


gc.collect()


# In[ ]:


data_df.to_hdf('optimized_data_df.hdf5', key='df', mode='w')


# In[20]:


data_df = pd.read_hdf('optimized_data_df.hdf5')


# In[285]:


len(data_df[data_df['failure'] == 1])


# В некотором смысле решающий момент - выборка из объединённого датафрейма данных которые будут переданы ML модели для обучения. Весь датафрейм передавать нет смысла - наличествует очень большой перекос (bias) в данных: 2263 строк в которых значение столбца failure равно 1 (из 40737546 строк). Сколько нибудь сложных методов отбора не используеться, для доказательства работоспособности методики это излишне. Выбираем саму "аварийную" строку и несколько предшествующих (в данном случае 3).

# In[127]:


start_time = time.time()
selected_data_df = select_failure_events_with_preceding(
    ordered_not_nans_data_df,
    preceding_min_subset_len=2,
    preceding_len_random=False
)
print("time_elapsed: {} min ".format((time.time() - start_time) / 60))


# Разбиваем данные на независмые переменные (features) X и зависимую (target) y.

# In[134]:


start_time = time.time()
X, y = divide_features_and_target(selected_data_df)
print("time_elapsed: {} min ".format((time.time() - start_time) / 60))


# Разделяем данные на обучающие и тестовые.

# In[135]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, shuffle=False)


# Для начала используем логистическую регрессию в качестве ML модели, далее её используем как базовую для сравнения с более сложными моделями.

# In[136]:


log_reg_model = LogisticRegression()


# In[137]:


log_reg_model.fit(X_train, y_train)


# Вычисляем prediction на тестовых данных.

# In[286]:


y_pred = log_reg_model.predict(X_test)


# Вычисляем accuracy.

# In[139]:


accuracy_score(y_test, y_pred)


# Как видим accuracy невелика, но лучше чем 0.5.

# Вычисляем confusion matrix.

# In[140]:


confusion_matrix(y_test, y_pred)


# Видим что верно предсказанных сбоев - 33, неверно предсказанных - 6, непредсказанных - 533, верно предсказанных "несбоев": 1125 из 1697 событий в тестовой выборке. Результат не очень хороший. Но во общем для логистической регресси это и предполагалось.

# ##### Далее строим Gradient Boosting модели. Будут использоваться простейшие настройки без тонкого тюнинга моделей.

# Используем XGBoost - очень хорошо зарекомендававшую библиотеку для построения Gradient Boosting моделей.

# Создаём pipeline, на случай если в дальнейшем нам надо будет ещё добвавить действия (например дополнительную обработку данных) до или после вызова XGBoost классификатора.

# In[147]:


xgb_pipeline = Pipeline([('xgb', XGBClassifier())])


# Задаём наборы параметров для XGB классификатора. Они будут использованы для поиска по сетке наилучешего набора параметров, с использованием GridSearchCV.

# In[148]:


xgb_param_grid = {
    'xgb__n_estimators': [10, 50, 100, 500],
    'xgb__learning_rate': [0.1, 0.5, 1]
}

xgb_fit_params = {
    'xgb__eval_set': [(X_test, y_test)],
    'xgb__early_stopping_rounds': 10,
    'xgb__verbose': 3
}


# In[149]:


xgb_search_cv = GridSearchCV(xgb_pipeline, cv=5, param_grid=xgb_param_grid)


# In[150]:


start_time = time.time()
xgb_search_cv.fit(X_train, y_train, **xgb_fit_params)
print("elapsed time: {} min".format((time.time() - start_time) / 60))


# Так же как и для логистической регресси находим prediction на тестовых данных.

# In[153]:


y_pred_xgb = xgb_search_cv.best_estimator_.predict(X_test)


# Вычисляем accuracy:

# In[154]:


accuracy_score(y_test, y_pred_xgb)


# Как видим accuracy уже значительно лучше.

# Вычисляем confusion matrix.

# In[155]:


confusion_matrix(y_test, y_pred_xgb)


# Как видим XGBoost верно предсказал 413 отказов, неверно предсказал 52 отказа, не предсказал 153, верно предсказал 1079 "неотказов" что уже можно использовать в "боевой" обстановке.

# Далее используем LightGBM - библиотеку соперничающюю с XGBoost в эффективности. Производим те же самые шаги что и для XGBoost.

# In[55]:


lgbm_pipeline = Pipeline([('lgbm', LGBMClassifier())])


# In[158]:


lgbm_param_grid = {
    'lgbm__n_estimators': [10, 50, 100, 500],
    'lgbm__learning_rate': [0.1, 0.5, 1]
}

lgbm_fit_params = {
    'lgbm__early_stopping_rounds': 30, 
    'lgbm__eval_metric': 'auc', 
    'lgbm__eval_set': [(X_test,y_test)],
    'lgbm__eval_names': 'valid',
    'lgbm__verbose': 100,
}


# In[159]:


lgbm_search_cv = GridSearchCV(lgbm_pipeline, cv=5, param_grid=lgbm_param_grid)


# In[160]:


start_time = time.time()
lgbm_search_cv.fit(X_train, y_train, **lgbm_fit_params)
print("elapsed time: {} min".format((time.time() - start_time) / 60))


# In[161]:


y_pred_lgbm = lgbm_search_cv.best_estimator_.predict(X_test)


# In[162]:


accuracy_score(y_test, y_pred_lgbm)


# Что ж accuracy относительно неплох и примерно равен accuracy для XGBoost.

# Кроме accuracy вычисляем roc auc.

# In[163]:


roc_auc_score(y_test, y_pred_lgbm)


# roc auc относительно неплох.

# In[164]:


confusion_matrix(y_test, y_pred_lgbm)


# confusion matrix выглядит чуть получше чем у XGBoost.

# Объединяем прдсказания XGBoost и LightGBM (ставим 1 там где хотя бы одна из моделей ставит 1)

# In[165]:


y_pred_xgb_lgbm = pd.Series(y_pred_xgb).combine(pd.Series(y_pred_lgbm), lambda u, v: u or v, fill_value=1)


# In[166]:


accuracy_score(y_test, y_pred_xgb_lgbm)


# In[167]:


roc_auc_score(y_test, y_pred_xgb_lgbm)


# In[168]:


confusion_matrix(y_test, y_pred_xgb_lgbm)


# Как видим посредством объединения удалось добиться некоторого улучшения.

# Так же небезинтересно было бы посмотреть на кривые обучения. Определяем вспомогательные функции для "рисования" соответствующих графиков.

# In[169]:


def plot_learning_curve(train_sizes, train_scores, test_scores, title, alpha=0.1, measure='accuracy'):
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.plot(train_sizes, train_mean, label='train score', color='blue', marker='o')
    
    plt.fill_between(
        train_sizes,
        train_mean + train_std,
        train_mean - train_std,
        color='blue',
        alpha=alpha
    )
    
    plt.plot(
        train_sizes,
        test_mean,
        label='test score',
        color='red',
        marker='o'
    )

    plt.fill_between(
        train_sizes,
        test_mean + test_std,
        test_mean - test_std,
        color='red',
        alpha=alpha
    )
    
    plt.title(title)
    plt.xlabel("Number of training points")
    plt.ylabel("Measure {}".format(measure))
    plt.grid(ls='--')
    plt.legend(loc='best')
    plt.show()


def plot_validation_curve(param_range, train_scores, test_scores, title, alpha=0.1):
    param_range = [x[1] for x in param_range] 
    sort_idx = np.argsort(param_range)
    param_range=np.array(param_range)[sort_idx]
    train_mean = np.mean(train_scores, axis=1)[sort_idx]
    train_std = np.std(train_scores, axis=1)[sort_idx]
    test_mean = np.mean(test_scores, axis=1)[sort_idx]
    test_std = np.std(test_scores, axis=1)[sort_idx]
    plt.plot(param_range, train_mean, label='train score', color='blue', marker='o')
    plt.fill_between(param_range, train_mean + train_std,
                 train_mean - train_std, color='blue', alpha=alpha)
    plt.plot(param_range, test_mean, label='test score', color='red', marker='o')
    plt.fill_between(param_range, test_mean + test_std, test_mean - test_std, color='red', alpha=alpha)
    plt.title(title)
    plt.grid(ls='--')
    plt.xlabel('Weight of class 2')
    plt.ylabel('Average values and standard deviation for F1-Score')
    plt.legend(loc='best')
    plt.show()


# In[170]:


plt.figure(figsize=(9, 6))


# In[171]:


cv = KFold(n_splits=5, random_state=42)


# In[172]:


xgb_train_sizes, xgb_train_scores, xgb_test_scores = learning_curve(
    estimator=xgb_search_cv.best_estimator_,
    X=X_train,
    y=y_train,
    train_sizes=np.arange(0.1, 1.1, 0.1),
    cv=cv,
    scoring='accuracy',
    n_jobs= 4
)


# In[173]:


plot_learning_curve(
    xgb_train_sizes,
    xgb_train_scores,
    xgb_test_scores,
    title='Learning curve for XGBoost'
)


# In[174]:


lgbm_train_sizes, lgbm_train_scores, lgbm_test_scores = learning_curve(
    estimator=lgbm_search_cv.best_estimator_,
    X=X_train,
    y=y_train,
    train_sizes=np.arange(0.1, 1.1, 0.1),
    cv=cv,
    scoring='roc_auc',
    n_jobs= 4
)


# In[246]:


plot_learning_curve(
    lgbm_train_sizes,
    lgbm_train_scores,
    lgbm_test_scores,
    title="Learning curve for LightGBM",
    measure="roc_auc"
)


# Как видим, кривые обучения пока далеки от совершенства.

# Этот Jupyter notebook всего лишь prof of concepts.
# В дальнейшем, используя более изощрённые методы борьбы с перекошенностью данных (сложные алгоритмы выбора, либо построения дополнитель точек отказа), используя более сложные модели (Neural Networks различных видов) и более тонко настраивая их скорее всего можно будет добиться очень сильного уменьшения ложных предсказаний/"непредсказаний" и увеличения правильных предсказаний.
# Так же имеет смысл рассмотреть эту задачу как anomaly detection задачу.

# In[288]:


data_df.columns.to_list()


# In[290]:


data_df.head(10).to_json('smart_short_0.json')


# In[ ]:




