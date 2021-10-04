#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 12:02:05 2021

@author: ronimalihi
"""


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn import decomposition #decomposition module
from pylab import rcParams
from pandas import DataFrame
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


rcParams['figure.figsize'] = 12, 4
sns.set_style('whitegrid')

np.random.seed(42)



aisles = pd.read_csv("/Users/ronimalihi/Desktop/project/t/aisles.csv")
departments = pd.read_csv("/Users/ronimalihi/Desktop/project/t/departments.csv")
prior = pd.read_csv("/Users/ronimalihi/Desktop/project/t/order_products__prior.csv")
train = pd.read_csv("/Users/ronimalihi/Desktop/project/t/order_products__train.csv")
orders = pd.read_csv("/Users/ronimalihi/Desktop/project/t/orders.csv")
products = pd.read_csv("/Users/ronimalihi/Desktop/project/t/products.csv")
customers = pd.read_csv("/Users/ronimalihi/Desktop/project/t/Mall_Customers.csv")
#Union of all the relevent tables

all_orders = pd.concat([prior,train],axis = 0)
all_orders = all_orders.merge(products[['product_id','aisle_id','department_id']], how = 'inner', on = 'product_id')
all_orders = all_orders.merge(aisles, on = 'aisle_id')
all_orders = all_orders.merge(departments, on = 'department_id')
all_orders = all_orders.merge(orders[['order_id','user_id']], on = 'order_id')
all_orders = all_orders.merge(customers[['user_id','Age','District']], on = 'user_id')



aisle_hist2 = all_orders[['user_id','add_to_cart_order','aisle']].groupby(['user_id','aisle']).sum().reset_index()
user_volume2 = aisle_hist2.groupby('user_id')['add_to_cart_order'].sum()
user_volume2 = user_volume2.reset_index().rename(columns = {'add_to_cart_order':'volume'})
aisle_hist2 = aisle_hist2.merge(user_volume2, how = 'inner', on = 'user_id')
aisle_hist2['aisle_share'] = aisle_hist2['add_to_cart_order'] / aisle_hist2['volume']

#customers_data=all_orders[['user_id','Age','Gender','District']].groupby(['user_id','Age','Gender','District']).sum().reset_index()
#customers_data=pd.get_dummies(customers_data[['Gender','District']])
#ages= all_orders[['user_id','Age']].groupby(['user_id','Age']).sum().reset_index()
check= all_orders[['user_id','District']].groupby(['user_id','District']).sum().reset_index()
check2= check.groupby('District')['user_id'].value_counts().rename('volume').reset_index()

check3=check.groupby('District')['user_id'].count().rename('city_count').reset_index()

check=check.merge(check2[['user_id','volume']],how ='inner', on= 'user_id')
check=check.merge(check3[['District','city_count']], on= 'District')

check['city']=check['city_count']/check['city_count'].count()

aisle_hist2= aisle_hist2.merge(check[['user_id','city']],on='user_id')
check=check.sort_values('user_id').set_index('user_id')
ages= all_orders[['user_id','Age']].groupby(['user_id','Age']).sum().reset_index(1)

Districts = all_orders[['user_id','District']].groupby(['user_id','District']).sum().reset_index(1)
#gender_list =  all_orders[['user_id','Gender']].groupby(['user_id','Gender']).sum().reset_index()
Districts_numeric = pd.get_dummies(Districts)
#gender_numeric = pd.get_dummies(gender_list)

aisle_vol_pivot = aisle_hist2[['user_id','aisle','add_to_cart_order']].pivot(index = 'user_id', columns = 'aisle', values = 'add_to_cart_order')
aisle_share_pivot = aisle_hist2[['user_id','aisle','aisle_share']].pivot(index = 'user_id', columns = 'aisle', values = 'aisle_share')
aisle_share_pivot['Age']=ages['Age']
aisle_share_pivot['District']=Districts['District']

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

aisle_share_pivot['District'] = le.fit_transform(aisle_share_pivot['District'])

y = le.transform(Districts)

aisle_share_pivot['District_Haifa']=Districts_numeric['District_Haifa']
aisle_share_pivot['District_Jerusalem']=Districts_numeric['District_Jerusalem']
aisle_share_pivot['District_Tel Aviv']=Districts_numeric['District_Tel Aviv']
aisle_share_pivot['District_North of Israel']=Districts_numeric['District_North of Israel']
aisle_share_pivot['District_Central Israel']=Districts_numeric['District_Central Israel']
aisle_share_pivot['District_South of Israel']=Districts_numeric['District_South of Israel']
aisle_share_pivot['Gender_Female']=gender_numeric['Gender_Female']
aisle_share_pivot['Gender_Male']=gender_numeric['Gender_Male']


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
# transform data
aisle_share_pivot= aisle_share_pivot.fillna(value = 0)

scaled = scaler.fit_transform(aisle_share_pivot)

aisle_share_pivot=pd.DataFrame(scaled, index=aisle_share_pivot.index, columns=aisle_share_pivot.columns)
scaler = MinMaxScaler()
cal=scaler.fit_transform(aisle_share_pivot)
aisle_share_pivot=pd.DataFrame(cal, index=aisle_share_pivot.index, columns=aisle_share_pivot.columns)

aisle_share_pivot['District_Haifa']=Districts_numeric['District_Haifa']
aisle_share_pivot['District_Jerusalem']=Districts_numeric['District_Jerusalem']
aisle_share_pivot['District_Tel Aviv']=Districts_numeric['District_Tel Aviv']
aisle_share_pivot['District_North of Israel']=Districts_numeric['District_North of Israel']
aisle_share_pivot['District_Central Israel']=Districts_numeric['District_Central Israel']
aisle_share_pivot['District_South of Israel']=Districts_numeric['District_South of Israel']


print(correct/len(trainX))
#aisle_share_pivot= pd.DataFrame(scaler.transform(aisle_share_pivot), index=aisle_share_pivot.index, columns=aisle_share_pivot.columns)

aisle_vol_pivot = aisle_vol_pivot.fillna(value = 0)
X = aisle_share_pivot.values

print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = list(aisle_share_pivot.columns)
for i in range(true_k):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind]),
    print
def cluster_predict(str_input):
    Y = aisle_share_pivot.transform(list(str_input))
    prediction = model.predict(Y)
    return prediction

# Continuing after vectorization step
# data-structure to store Sum-Of-Square-Errors
sse = {}
labels={}
# Looping over multiple values of k from 1 to 30
for k in range(1, 13):
    model = KMeans(n_clusters=k, init='k-means++', max_iter=100).fit(X)
    
    sse[k] = model.inertia_
    labels[k] = model.labels_
# Plotting the curve with 'k'-value vs SSE
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
# Save the Plot in current directory
plt.savefig('elbow_method.png')

cluster_df = pd.concat([X[[0,1,2,3,4,5]],pd.Series(labels[6]).rename('cluster')], axis = 1)


ks = range(1,13) #hit and trial, let's try it 10 times.
inertias = []
for k in ks:
    model = KMeans(n_clusters=k)
    model.fit(X)
   # Create a KMeans instance with k clusters: model
                       # Fit model to samples
    inertias.append(model.inertia_) # Append the inertia to the list of inertias

plt.plot(ks, inertias, '-o', color='black') #Plotting. The plot will give the 'elbow'.
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()

k_means = KMeans(n_clusters=5)
#Run the clustering algorithm
model = k_means.fit(X)
model
k_means.cluster_centers_
k_means.inertia_
labels = k_means.labels_

# check how many of the samples were correctly labeled
correct_labels = sum(y == labels)

print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))
print('Accuracy score: {0:0.2f}'. format(correct_labels/float(y.size)))
#Generate cluster predictions and store in y_hat
y_hat = k_means.predict(X)
from sklearn import metrics
labels = k_means.labels_
metrics.silhouette_score(X, labels, metric = 'euclidean')
metrics.calinski_harabasz_score(X, labels)
labels = k_means.labels_

# check how many of the samples were correctly labeled
correct_labels = sum(y == labels)

print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))
# data-structure
kmeans = KMeans(n_clusters=6)
clusters= kmeans.fit_transform(X)

pred = kmeans.fit_predict(X)

t=kmeans.fit_predict(clusters)
cluster_df = pd.DataFrame(clusters)
pca = decomposition.PCA(n_components=10)
pca_user_order = pca.fit_transform(X)

kmeans.explained_variance_ratio_

label_color_mapping = {0:'r', 1: 'g', 2: 'b',3:'c' , 4:'m', 5:'y'}
label_color = [label_color_mapping[l] for l in pred]

#Scatterplot showing the cluster to which each user_id belongs.
plt.figure(figsize = (15,8))
plt.scatter(clusters[:,0],clusters[:,2], c= label_color, alpha=0.3) 
plt.xlabel = 'X-Values'
plt.ylabel = 'Y-Values'
plt.show()

cluster_df = pd.concat([cluster_df[[0,1,2,3,4,5]],pd.Series(labels[6]).rename('cluster')], axis = 1)


frame = pd.DataFrame(X)
cluster_df['cluster'] = pred
frame['cluster']= pred
cluster_df = pd.concat([pd.Series(aisle_share_pivot.index),cluster_df],axis=1)

sns.pairplot(cluster_df, hue = 'cluster')


cluster_aisle_br = aisle_hist2.merge(cluster_df[['user_id','cluster']], on = 'user_id')

cluster_aisle_br = cluster_aisle_br.rename(columns = {'aisle_share':'user_aisle_share'})


aisle_hist2 = aisle_hist2.merge(cluster_df[['user_id','cluster']], on = 'user_id')
aisle_hist2 = aisle_hist2.rename(columns = {'aisle_share':'user_aisle_share'})
cluster_aisle_br = pd.DataFrame(columns = ['aisle','add_to_cart_order','aisle_share','cluster'])
cluster_aisle_br.to_csv('/Users/ronimalihi/Desktop/project/t/clusters_aisle_br.csv')
aisle_hist2.to_csv('/Users/ronimalihi/Desktop/project/t/aisle_hist2.csv')

for i in range(0,6):
    x = aisle_hist2[aisle_hist2['cluster'] == i]
    x = x.groupby('aisle')['add_to_cart_order'].sum().reset_index()
    x['aisle_share'] = x['add_to_cart_order']
    x['aisle_share'] = x['aisle_share'].apply(lambda f: f / x['add_to_cart_order'].sum())
    x['cluster'] = i
    cluster_aisle_br = pd.concat([x,cluster_aisle_br], axis = 0)

t20 = list(aisle_hist2.groupby('aisle')['add_to_cart_order'].sum().sort_values(ascending = False)[0:20].index)
t30 = list(aisle_hist2.groupby('aisle')['add_to_cart_order'].sum().sort_values(ascending = False)[0:30].index)

aisle_heat = cluster_aisle_br[cluster_aisle_br['aisle'].isin(t30)]
aisle_heat = aisle_heat.pivot(index = 'aisle', columns = 'cluster', values = 'aisle_share')
sns.heatmap(aisle_heat,cmap="YlGnBu")

all_orders = all_orders.merge(cluster_df[['user_id','cluster']], on = 'user_id')
all_orders.to_csv('/Users/ronimalihi/Desktop/project/t/all_orders.csv')

aislest= aisle_hist2.groupby('aisle')['user_id'].count().reset_index()
cluster_aisle_br=cluster_aisle_br.merge(aisles,on=['aisle'])
aisle_hist2.to_csv('/Users/ronimalihi/Desktop/project/t/aisle_hist2.csv')
cluster_df.to_csv('/Users/ronimalihi/Desktop/project/t/clusters_df.csv')
all_orders = all_orders.merge(cluster_df[['user_id','cluster']], on = 'user_id')



cluster_aisle_br.groupby('cluster')[['aisle','aisle_id']].count()   
#association rule

df2 = all_orders.merge(products[['product_id','product_name']], how = 'inner', on = 'product_id')
df2=df2.index_col= 0
df2= df2.sort_values('order_id')



df2.dropna(axis=0, subset=['order_id'], inplace=True)
df2['order_id'] = df2['order_id'].astype('str')
#Checking with only a few samples. Concept is replicable.
np.random.seed(972) # set the seed to make examples repeatable
df2 = df2.sample(n=2000)[['user_id','product_name','cluster']]
basket = pd.crosstab(df2['user_id'],df2['product_name']).astype('bool').astype('int')
basket=basket.reset_index(drop=True)
basket.index
del df2
basket = (df2[df2['cluster'] == 5]
          .groupby(['order_id', 'product_id'])['add_to_cart_order']
          .sum().unstack().reset_index().fillna(0)
          .set_index('order_id')).astype('bool').astype('int')

import mlxtend.preprocessing
import mlxtend.frequent_patterns

import matplotlib.pyplot as plt
cluster_orders = {}
for x in range(0,6):
    i = df2[df2['cluster'] == x]
    i = i.set_index('user_id')['product_name'].rename('item_id')
    cluster_orders[x] = i
    
online_encoder = mlxtend.preprocessing.TransactionEncoder()
online_encoder_array = online_encoder.fit_transform(cluster_orders[5])

test2 = pd.DataFrame(cluster_orders[2]).reset_index()
tt= pd.DataFrame(test2.columns,test2)
te1= pd.crosstab(test2['order_id'],test2['item_id']).astype('bool').astype('int')
te1=te1.reset_index(drop=True)
te1.index


    group_association_rules_dic = {}
for x in range(0,6):
    group_association_rules_dic[x] = apriori (cluster_orders[x],min_support=0.001,use_colnames=True)
    
    
df2 = df2[['order_id','product_name']]
basket = pd.crosstab(df2['order_id'],df2['product_name']).astype('bool').astype('int')
del df2

df= df2[['use_id','product_name']]
l=df2[['user_id','product_name']]
basket = (df2.groupby(['order_id', 'product_id'])['add_to_cart_order'].sum().unstack().reset_index().fillna(0).set_index('order_id'))
basket = pd.crosstab(df2['order_id'],df2['product_name'])
basket=basket.reset_index(drop=True)
basket.index
def encode_units(m):
    if m <= 0:
        return 0
    if m >= 1:
        return 1

basket_sets = basket.applymap(encode_units)
basket_sets.drop('POSTAGE', inplace=True, axis=1)


group_association_rules_dic = {}
for x in range(0,6):
    group_association_rules_dic[x] = apriori(cluster_orders[x],.0001)
    
frequent_itemsets=apriori(basket, min_support=0.00002, use_colnames=True).sort_values('support', ascending=False) 
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules.head()
frequent_itemsets_ap = apriori(basket, min_support=0.00002, use_colnames=True).sort_values('support', ascending=False) 
frequent_itemsets_ap.head(20)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
frequent_itemsets[frequent_itemsets['length'] >= 2]
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules[(rules['lift'] >= 5) & (rules['confidence']>= 0.5)] 

rules.head()

cluster_aisle_orders = {}
for x in range(0,6):
    i = df2[df2['cluster'] == x]
    i = i.groupby(['order_id','aisle_id'])['product_id'].count().reset_index().set_index('order_id')['aisle_id'].rename('item_id')
    cluster_aisle_orders[x] = i

print("Top terms per cluster:")
order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
terms = list(aisle_share_pivot[:-1].columns)
for i in range(4):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :15]:
        print(' %s' % terms[ind]),
    print
cluster_item_rules_dic = {}
cluster_item_rules_dic[0] = cluster0_item_rules
cluster_item_rules_dic[1] = cluster1_item_rules
cluster_item_rules_dic[2] = cluster2_item_rules
cluster_item_rules_dic[3] = cluster3_item_rules
cluster_item_rules_dic[4] = cluster4_item_rules
cluster_item_rules_dic[5] = cluster5_item_rules

for i in range(0,6):
    x = sns.barplot(data = cluster_aisle_br[cluster_aisle_br['cluster'] == i].sort_values('aisle_share', ascending = False)[0:10], x = 'aisle', y = 'aisle_share')
    x.set_xticklabels(x.get_xticklabels(), rotation=90)
    plt.title(str(i))
    x.set(xlabel = 'Aisle', ylabel = 'Aisle Share')
    plt.figure()
cluster_aisle_br = aisle_hist2.merge(frame[['user_id','cluster']], on = 'user_id')
pca = decomposition.PCA(n_components=3)
pca_user_order = pca.fit_transform(X)
pca.explained_variance_ratio_.sum()
# returns recommended products given inputs
clusters = {}
    n = 0
    for item in pred:
        if item in clusters:
            clusters[item].append(row_dict[n])
        else:
            clusters[item] = [row_dict[n]]
        n +=1
features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_ratio_, color='black')
plt.xlabel('PCA features')
plt.ylabel('variance %')
all_orders = all_orders.merge(cluster_df[['user_id','cluster']], on = 'user_id')
import pandas as pd
import numpy as np

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
all_orders.to_csv('all_orders.csv')
 test = all_orders = all_orders.merge(products[['product_id','product_name']], how = 'inner', on = 'product_id')
np.random.seed(942) # set the seed to make examples repeatable
df2 = test.sample(n=1000)[['user_id','product_name']]
basket = pd.crosstab(df2['user_id'],df2['product_name']).astype('bool').astype('int')
basket=basket.reset_index(drop=True)

frequent_itemsets=apriori(basket, min_support=0.00002, use_colnames=True).sort_values('support', ascending=False) 

#############

def main():
    
    

          
    st.title("Simple Login App")

    menu = ["Home","Login","SignUp"]
    choice = st.sidebar.selectbox("Menu",menu)
    

    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    if choice == "Home":
        st.subheader("Home")
        cluster_aisle_br=load_data("t/clusters_aisle_br.csv")
        aisle_hist2=load_data("t/aisle_hist2.csv")#*#*#
        for i in range(0,6):
            x = aisle_hist2[aisle_hist2['cluster'] == i]
            x = x.groupby('aisle')['add_to_cart_order'].sum().reset_index()
            x['aisle_share'] = x['add_to_cart_order']
            x['aisle_share'] = x['aisle_share'].apply(lambda f: f / x['add_to_cart_order'].sum())
            x['cluster'] = i
            cluster_aisle_br = pd.concat([x,cluster_aisle_br], axis = 0)
        
        t20 = list(aisle_hist2.groupby('aisle')['add_to_cart_order'].sum().sort_values(ascending = False)[0:20].index)
        t30 = list(aisle_hist2.groupby('aisle')['add_to_cart_order'].sum().sort_values(ascending = False)[0:30].index)
        
        aisle_heat = cluster_aisle_br[cluster_aisle_br['aisle'].isin(t20)]
        aisle_heat = aisle_heat.pivot(index = 'aisle', columns = 'cluster', values = 'aisle_share')
        fig, ax = plt.subplots()
        sns.heatmap(aisle_heat,cmap="YlGnBu")
        st.write(fig)
        
        
                
                
        for i in range(0,6):
            x = sns.barplot(data = cluster_aisle_br[cluster_aisle_br['cluster'] == i].sort_values('aisle_share', ascending = False)[0:10], x = 'aisle', y = 'aisle_share')
            x.set_xticklabels(x.get_xticklabels(), rotation=90)
            plt.title(str(i))
            x.set(xlabel = 'Aisle', ylabel = 'Aisle Share')
            plt.figure()

            fig, ax = plt.subplots()
           
            st.pyplot(fig,ax)
        



    elif choice == "Login":
        st.subheader("Login Section")


        username = st.sidebar.text_input("User Name")
        password = st.sidebar.text_input("Password",type='password')
        if st.sidebar.checkbox("Login"):
            # if password == '12345':
            create_usertable()
            hashed_pswd = make_hashes(password)

            result = login_user(username,check_hashes(password,hashed_pswd))
       
            if result:
                st.success("Logged In as {}".format(username))
                task = st.selectbox("Task",["Clusters Data","Recommend","Profiles"])
                df= load_data("t/clusters_df.csv")



                if task == "Clusters Data":
                    st.subheader("")
                    
                    #with st.beta_expander("Title"):
                     #   mytext= st.text_area("Type Here")
                      #  st.write(mytext)
                        
                    products_list= df['cluster'].drop_duplicates().tolist()
                    product_choice= st.selectbox("Clusters", products_list)
                    with st.expander('Products',expanded=False):
                        st.dataframe(df.head(10))
                        
                        title_link= df[df['cluster']==product_choice]['cluster'].values
                        pro_link= df[df['cluster']==product_choice]['user_id'].values
                    s1,s2,s3= st.beta_columns(3)
                    
                    with s1:
                        with st.expander("Cluster",expanded=False):
                            st.success(title_link)

                    with s2:
                        with st.expander("Users",expanded=False):
                            st.write(pro_link)
                            
                            
                    with s3:
                        with st.form(key= "Email Form"):
                            email= st.text_input("Email")
                            
                            submit_email= st.form_submit_button(label='subscribe')
                            if submit_email:
                                try:
                                    connection= s.SMTP(smtp.gmail.com,587)
                                    connection.starttls()
                                st.success(" A message was sent to {}".format(email))

                elif task == "Recommend":
                    st.subheader("Analytics")
                    cosine_sim= vectorize_text(df['aisle'])  
                    search_term= st.text_input("search")
                    num_of_rec= st.sidebar.number_input("Number",4,20,7)
                    if st.button("Recommend"):
                        if search_term is not None:
                            resu=get_recommendation(search_term,cosine_sim, df, num_of_rec)
                            st.write(resu) 
                    # data= pd.read_csv('all_orders.csv', index_col = 0)
                   # t=pd.DataFrame(data)
                    #st.dataframe(t)
                elif task == "Profiles":
                    st.subheader("User Profiles")
                    user_result = view_all_users()
                    clean_db = pd.DataFrame(user_result,columns=["Username","Password"])
                    st.dataframe(clean_db)
            else:
                st.warning("Incorrect Username/Password")

    elif choice == "SignUp":
        st.subheader("Create New Account")
        new_user = st.text_input("Username",key='1')
        new_password = st.text_input("Password",type='password',key='2')

        if st.button("Signup"):
            create_usertable()
            add_userdata(new_user,make_hashes(new_password))
            st.success("You have successfully created a valid Account")
            st.info("Go to Login Menu to login")
            

            
if __name__ =='__main__':
    main()
