#%% Importálás
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import scipy.stats as st
from datetime import datetime
import seaborn as sns
#%% Fájlok beolvasása
ldir = os.chdir(r"C:\Users\tothd\OneDrive\Asztali gép\Egyetem dolgok\Számítógépes adatelemzés")

GBdata = pd.read_csv('GBvideos.csv', parse_dates=['publish_time'], usecols=['video_id', 'trending_date', 'title', 'channel_title', 'category_id', 'publish_time', 'views', 'likes', 'dislikes', 'comment_count', 'comments_disabled'])
DEdata = pd.read_csv('DEvideos.csv', parse_dates=['publish_time'], usecols=['video_id', 'trending_date', 'title', 'channel_title', 'category_id', 'publish_time', 'views', 'likes', 'dislikes', 'comment_count', 'comments_disabled'])
FRdata = pd.read_csv('FRvideos.csv', parse_dates=['publish_time'], usecols=['video_id', 'trending_date', 'title', 'channel_title', 'category_id', 'publish_time', 'views', 'likes', 'dislikes', 'comment_count', 'comments_disabled'])
CAdata = pd.read_csv('CAvideos.csv', parse_dates=['publish_time'], usecols=['video_id', 'trending_date', 'title', 'channel_title', 'category_id', 'publish_time', 'views', 'likes', 'dislikes', 'comment_count', 'comments_disabled'])
INdata = pd.read_csv('INvideos.csv', parse_dates=['publish_time'], usecols=['video_id', 'trending_date', 'title', 'channel_title', 'category_id', 'publish_time', 'views', 'likes', 'dislikes', 'comment_count', 'comments_disabled'])
USdata = pd.read_csv('USvideos.csv', parse_dates=['publish_time'], usecols=['video_id', 'trending_date', 'title', 'channel_title', 'category_id', 'publish_time', 'views', 'likes', 'dislikes', 'comment_count', 'comments_disabled'])

#%% df alakítás

USdata['country']='US'
CAdata['country']='CA'
GBdata['country']='GB'
DEdata['country']='DE'
FRdata['country']='FR'
INdata['country']='IN'

data=USdata.append([CAdata,GBdata,DEdata,FRdata,INdata])
data.shape

#%% A nap melyik órájában tették közzé a legtöbb videót. US

data['publish_hour'] = data['publish_time'].dt.hour
dataUS = data.loc[data['country'] == 'US']

hours,counts = zip(*sorted(dataUS.publish_hour.value_counts().to_dict().items(),key=lambda val:val[0]))

fig,ax = plt.subplots(figsize=(10,6))

cmap = plt.get_cmap('inferno')

colors=[cmap(i) for i in np.linspace(0, 1, len(hours))]

ax.set_title('United States')
ax.bar(hours,counts,color=colors)
ax.set_xticks(range(len(hours)))
ax.set_xticklabels(hours)
ax.set_xlabel('Hour of a day')
ax.set_ylabel('Number of videos published');

#%% GB

dataGB = data.loc[data['country'] == 'GB']

hours,counts = zip(*sorted(dataGB.publish_hour.value_counts().to_dict().items(),key=lambda val:val[0]))

fig,ax=plt.subplots(figsize=(10,6))

cmap = plt.get_cmap('magma')
colors=[cmap(i) for i in np.linspace(0, 1, len(hours))]

ax.set_title('Great Britain')
ax.bar(hours,counts,color=colors)
ax.set_xticks(range(len(hours)))
ax.set_xticklabels(hours)
ax.set_xlabel('Hour of a day')
ax.set_ylabel('Number of videos published');

#%% FR
dataFR = data.loc[data['country'] == 'FR']

hours,counts = zip(*sorted(dataFR.publish_hour.value_counts().to_dict().items(),key=lambda val:val[0]))

fig,ax=plt.subplots(figsize=(10,6))

cmap = plt.get_cmap('spring')
colors=[cmap(i) for i in np.linspace(0, 1, len(hours))]

ax.set_title('France')
ax.bar(hours,counts,color=colors)
ax.set_xticks(range(len(hours)))
ax.set_xticklabels(hours)
ax.set_xlabel('Hour of a day')
ax.set_ylabel('Number of videos published');


#%%DE
dataDE = data.loc[data['country'] == 'DE']

hours,counts = zip(*sorted(dataDE.publish_hour.value_counts().to_dict().items(),key=lambda val:val[0]))

fig,ax=plt.subplots(figsize=(10,6))

cmap = plt.get_cmap('twilight')
colors=[cmap(i) for i in np.linspace(0, 1, len(hours))]

ax.set_title('Germany')
ax.bar(hours,counts,color=colors)
ax.set_xticks(range(len(hours)))
ax.set_xticklabels(hours)
ax.set_xlabel('Hour of a day')
ax.set_ylabel('Number of videos published');
#%%CA
dataCA = data.loc[data['country'] == 'CA']

hours,counts = zip(*sorted(dataCA.publish_hour.value_counts().to_dict().items(),key=lambda val:val[0]))

fig,ax=plt.subplots(figsize=(10,6))

cmap = plt.get_cmap('rainbow')
colors=[cmap(i) for i in np.linspace(0, 1, len(hours))]

ax.set_title('Canada')
ax.bar(hours,counts,color=colors)
ax.set_xticks(range(len(hours)))
ax.set_xticklabels(hours)
ax.set_xlabel('Hour of a day')
ax.set_ylabel('Number of videos published');
#%%IN

dataIN = data.loc[data['country'] == 'IN']

hours,counts = zip(*sorted(dataIN.publish_hour.value_counts().to_dict().items(),key=lambda val:val[0]))

fig,ax=plt.subplots(figsize=(10,6))

cmap = plt.get_cmap('cividis')
colors=[cmap(i) for i in np.linspace(0, 1, len(hours))]

ax.set_title('India')
ax.bar(hours,counts,color=colors)
ax.set_xticks(range(len(hours)))
ax.set_xticklabels(hours)
ax.set_xlabel('Hour of a day')
ax.set_ylabel('Number of videos published');


#%% scatterplot

plt.figure(figsize=(8,8))
sns.scatterplot(x='views',y='likes',data=data,hue='country',alpha=.8,palette='rainbow')

#%% heatmap
import seaborn as sns
sns.heatmap(data.corr(), annot = True)

#%% plot by categories

group_cat = data[['category_id','video_id']].groupby('category_id').count()
group_cat2 = data[['category_id','views']].groupby('category_id').sum()
group_cat = group_cat.join(group_cat2)
group_cat['video_id_percent'] = group_cat['video_id']/group_cat['video_id'].sum()
group_cat['views_percent'] = group_cat['views']/group_cat['views'].sum()
group_cat

category = pd.read_csv('category_id.csv', header=None)
category = category.rename(columns={0:'category_id',1:'category_name'})
group_cat = group_cat.join(category.set_index('category_id'), lsuffix='_2')
group_cat

plt.rcParams['figure.facecolor'] = 'green'
group_cat[['video_id_percent','views_percent','category_name']].set_index('category_name').plot(kind='bar',width = 0.8,figsize=(16,8))


#%% Neurális háló

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

nn = data.drop(columns=['category_id','video_id','trending_date','channel_title','publish_time','title','comments_disabled','country'])
dataset = nn.values
nn.dropna()
dataset

X = dataset[:, 1:4]
Y = dataset[:, 0]
#X = input, Y = output columns
X

# define base model
def baseline_model():
    model = Sequential()
    model.add(Dense(3, input_dim=3, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# evaluate model
estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0)
kfold = KFold(n_splits=10)
results = cross_val_score(estimator, X, Y, cv=kfold)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

import math
print("RMSE: %.2f" % abs(results.mean())**(1/2))