
# coding: utf-8

# In[125]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn import metrics
import scipy
import sys
print(sys.version)


# In[28]:


games=pd.read_csv("E:/sem1/CS989/assignment/datasetass/Video_games.csv")


# In[29]:


type(games)


# In[30]:


games.shape


# In[31]:


games.columns


# In[32]:


games.head()


# In[74]:


#Ygames is to keep only the games after 2000-2016

Ygames=games[games['Year_of_Release']>=2000]


# In[75]:


Ygames.shape


# In[108]:


Ygames.head(3)


# In[107]:


Ygames.columns
#usercrits has null 
#devpubs has nulls

Ygames.isnull().sum()


# In[114]:





# In[77]:


#Sales of the games with their names 
# sales is in number million units sold 
salesgame=Ygames[['NA_Sales','EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']]

#genre-wise grouped games with their sales
genre=Ygames[['Genre','NA_Sales','EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']]

#user and critics scores to analyse the highest rated genre and game
usercrits=Ygames[['Global_Sales','Critic_Score','User_Score']]

#developer and publisher stats
devpubs=Ygames[['Developer','Publisher','NA_Sales','EU_Sales','JP_Sales','Other_Sales','Global_Sales']]

#platform stats for games
platform=Ygames[['Platform','Name','Genre','NA_Sales','EU_Sales','JP_Sales','Other_Sales','Global_Sales']]


# In[78]:


#sales analysis
from pandas.tools.plotting import scatter_matrix 

#Summary stats

fig, ax=plt.subplots()
ax.scatter(genre['Global_Sales'],genre['Genre'],alpha=0.6,color='red',marker='*')
plt.xlabel('Numbers')
plt.ylabel('Genre')
plt.title("Plot for Genre vs Global Sales")
plt.show()

plt.figure(figsize=(8,5))
plt.scatter(Ygames['Year_of_Release'],Ygames['Global_Sales'],alpha=0.6,color='g',marker='^')
plt.xlabel('Year')
plt.ylabel('Numbers')
plt.title('Scatter plot for Year vs Global Sales')
plt.show()


# In[79]:


#sales analysis-genre

#global sales bar chart
plt.figure(figsize=(10,5))
plt.barh(Ygames['Genre'],Ygames['Global_Sales'],align='center',alpha=0.6)
plt.xlabel('Global Sales')
plt.ylabel('Genre')
plt.title('Genre VS Sales')


# In[122]:


#global sales bar chart
plt.figure(figsize=(10,5))
plt.bar(Ygames['Year_of_Release'],Ygames['Global_Sales'],align='center',alpha=0.6,color='r')
plt.xlabel("Year")
plt.ylabel("Millions copies sold")
plt.show()


# In[81]:


plt.figure(figsize=(10,6))
Ygames["Genre"].value_counts().plot.bar()
plt.xlabel('Genre')
plt.ylabel('Number of games released')
plt.title('Number of games in every genre')
plt.show()


# In[82]:


plt.figure(figsize=(12,8))
Ygames['Year_of_Release'].value_counts().sort_values(ascending=False).plot.barh()

plt.title('Number of games Released in each year')
plt.xlabel('Number of games')
plt.ylabel('Year')
plt.show()


# In[83]:


plt.figure(figsize=(12,6))
Ygames["Platform"].value_counts().plot.bar()
plt.title('Number of games available on every platform')
plt.xlabel('Platforms')
plt.ylabel('Number of games')
plt.show()


# In[84]:


#Games distrbution in two decades
y1=Ygames=games[games['Year_of_Release']>2009]
plt.figure(figsize=(10,6))
y1["Platform"].value_counts().plot.bar()
plt.title('Number of games available on every platform during 2009-2016')
plt.xlabel('Platforms')
plt.ylabel('Number of games')
plt.show()
plt.figure(figsize=(10,6))
y1["Genre"].value_counts().plot.bar()
plt.title('Number of games available in different genres during 2009-2016')
plt.xlabel('genre')
plt.ylabel('Number of games')
plt.show()


# In[115]:


#gaming analysis before 2009
y2=Ygames=games[games['Year_of_Release']<=2008]
plt.figure(figsize=(11,6))
y2["Platform"].value_counts().plot.bar()
plt.title('Number of games available on every platform during 2000-2009')
plt.show()

plt.figure(figsize=(11,6))
y2["Genre"].value_counts().plot.bar()
plt.title('Number of games available in every genre during 2000-2009')
plt.xlabel('genre')
plt.ylabel('Number of games')
plt.show()


# In[86]:


plt.figure(figsize=(10,6))
Ygames['Year_of_Release'].plot.density()
plt.xlabel('Years')
plt.title('Density graph for Games and their year of release')
plt.show()


# In[87]:


#heat map
plt.figure(figsize=(8,6))
corr=Ygames.corr()
sns.heatmap(corr)
plt.show()


# In[88]:



plt.figure(figsize=(12,6))
games['Platform'].value_counts(sort=True,ascending=True).plot.bar()
plt.title('Number of Game Releases by Platform')
plt.ylabel('Number of Games')
plt.xlabel('Platform')


# In[118]:


#trends for sales

JAPAN=games.groupby('Year_of_Release').sum()['JP_Sales']
NA=games.groupby('Year_of_Release').sum()['NA_Sales']
EU=games.groupby('Year_of_Release').sum()['EU_Sales']
Other=games.groupby('Year_of_Release').sum()['Other_Sales']

#Other sales
Other.plot.bar(x=Other,y='Other_Sales',figsize=(12,6))
plt.title('Rest of the world sales per Year in Million Units')
plt.xlabel('Year of release')
plt.ylabel('Sales in Millions')


# In[90]:


#EU sales
EU.plot.bar(x=EU,y='EU_Sales',figsize=(12,6))
plt.title('EU sales per Year in Million Units')
plt.xlabel('Year of release')
plt.ylabel('Sales in Millions')


# In[91]:


#NA sales
NA.plot.bar(x=NA.index,y='NA_Sales',figsize=(12,6))
plt.title('NA sales per Year in Million Units')
plt.ylabel('Sales in Millions')


# In[92]:


#japan sales
japan.plot.bar(x=japan.index,y='JP_Sales',figsize=(12,6))
plt.title('Japan sales per Year in Million Units')
plt.ylabel('Sales in Millions')


# In[93]:


#predict gAmes
fig, ax = plt.subplots()
ax.scatter(x = Ygames['Critic_Score'], y = Ygames['Global_Sales'])
plt.ylabel('Global_Sales')
plt.xlabel('Critic_Score')
plt.show()


# In[94]:


#drop null values
ucr = usercrits.dropna()


# In[95]:


ucri = ucr.astype(int)

ucri.isnull().any()
ucrid = ucri.drop(ucri[(ucri['Critic_Score']>60) & (ucri['Global_Sales']>60)].index)


# In[96]:


#Linear regression
from sklearn.linear_model import LinearRegression

X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(ucrid,ucrid.Global_Sales,test_size=0.30)


# In[97]:


lm=LinearRegression()
lm.fit(X_train,Y_train)


# In[98]:


lm.fit(ucrid, ucrid.Global_Sales)


# In[99]:


print(lm.intercept_)
print(lm.coef_)


# In[57]:


pd.DataFrame(list(zip(ucrid.columns, lm.coef_)), columns = ['Features', "Coefficients"])


# In[58]:


plt.scatter(ucrid.Critic_Score,ucrid.Global_Sales, color='black',alpha=0.6)
plt.xlabel("Critic Score")
plt.ylabel("Global sales")
plt.title('Relationship between critics rating and Global sales')
plt.show()


# In[59]:


predict = lm.predict(X_test)
print(metrics.mean_squared_error(Y_test, predict))


# In[60]:


#Unsupervised

U=ucrid
U
plt.scatter(U.iloc[:,1],U.iloc[:,0])
plt.xlabel("Critic Scores")
plt.ylabel("Global sales ")
plt.title("Relationship between critic scores and global sales")
plt.show()

##Graph
# In[68]:


#Heirarchical Clustering
from sklearn import cluster
from sklearn.preprocessing import scale
data = scale(U)
n_samples, n_features = U.shape
n_digits = len(np.unique(U.Critic_Score))
model = cluster.AgglomerativeClustering(n_clusters=n_digits, linkage='average', affinity='cosine')
model.fit(data)


x=U.iloc[:,1]
y=U.iloc[:,0]

from sklearn.cluster import AgglomerativeClustering
for k in range(2, 21):
    model = AgglomerativeClustering(n_clusters=k).fit(U)
    labels = model.labels_
    print(k, metrics.calinski_harabaz_score(U, labels))
    
x=U.iloc[:,1]
y=U.iloc[:,0]

import numpy as np
from sklearn.cluster import AgglomerativeClustering
for k in range(2, 21):
    model = AgglomerativeClustering(n_clusters=k).fit(U)
    labels = model.labels_
    print(k, metrics.silhouette_score(U, labels))


# In[62]:


from scipy.cluster.hierarchy import dendrogram, linkage
model = linkage(data, 'ward')
plt.figure(figsize=(11,6))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Index sample')
plt.ylabel('distance')
dendrogram(model,truncate_mode='lastp', p=12, show_leaf_counts=False, leaf_rotation=90., leaf_font_size=8.,show_contracted=True,)
plt.show()


# In[63]:


from scipy.cluster.hierarchy import dendrogram, linkage
model = linkage(data, "ward")
plt.figure()
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(model, leaf_rotation=90., leaf_font_size=8.,)
plt.show()

