#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 10:59:59 2021

@author: ronimalihi
"""
import streamlit as st
import sqlite3
import pandas as pd
import smtplib as s
# DB Management
conn = sqlite3.connect('data.db')
c = conn.cursor()




import hashlib
def make_hashes(password):
	return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password,hashed_text):
	if make_hashes(password) == hashed_text:
		return hashed_text
	return False

# You can also use the verify functions of the various libraries for the same purpose

def create_usertable():
	c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)')


def add_userdata(username,password):
	c.execute('INSERT INTO userstable(username,password) VALUES (?,?)',(username,password))
	conn.commit()

def login_user(username,password):
	c.execute('SELECT * FROM userstable WHERE username =? AND password = ?',(username,password))
	data = c.fetchall()
	return data

def view_all_users():
    c.execute('SELECT * FROM userstable')
    data = c.fetchall()
    return data

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity,linear_kernel

def load_data(data):
    df=pd.read_csv(data)
    return df

def vectorize_text(data):
    count_vect = CountVectorizer()
    cv_mat= count_vect.fit_transform(data)
    cosine_sim= cosine_similarity(cv_mat) 
    return cosine_sim

#recommendation System
def get_recommendation(title,cosine_sim,df,num_of_rec=5):
    indicates= pd.Series(df.index,index=df["aisle"]).drop_duplicates()
    idx= indicates[title]
    sim_score=list(enumerate(cosine_sim[idx]))
    sim_score=sorted(sim_score,key=lambda x: x[1],reverse=True)
    return sim_score[1:]


def main():
    
    

          
    st.title("Simple Login App tt")

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
                               
                st.markdown(
                    '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.5.3/dist/css/bootstrap.min.css" integrity="sha384-TX8t27EcRE3e/ihU7zmQxVncDAy5uIKz4rEkgIXeMed4M0jlfIDPvg6uqKI2xXr2" crossorigin="anonymous">',
                    unsafe_allow_html=True,
                )
                query_params = st.experimental_get_query_params()
                tabs = ["Home", "About", "Contact"]
                if "tab" in query_params:
                    active_tab = query_params["tab"][0]
                else:
                    active_tab = "Home"
                
                if active_tab not in tabs:
                    st.experimental_set_query_params(tab="Home")
                    active_tab = "Home"
                
                li_items = "".join(
                    f"""
                    <li class="nav-item">
                        <a class="nav-link{' active' if t==active_tab else ''}" href="/?tab={t}">{t}</a>
                    </li>
                    """
                    for t in tabs
                )
                tabs_html = f"""
                    <ul class="nav nav-tabs">
                    {li_items}
                    </ul>
                """
                
                st.markdown(tabs_html, unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                
                if active_tab == "Home":
                    st.write("Welcome to my lovely page!")
                    st.write("Feel free to play with this ephemeral slider!")
                    st.slider(
                        "Does this get preserved? You bet it doesn't!",
                        min_value=0,
                        max_value=100,
                        value=50,
                    )
                elif active_tab == "About":
                    st.write("This page was created as a hacky demo of tabs")
                elif active_tab == "Contact":
                    st.write("If you'd like to contact me, then please don't.")
                else:
                    st.error("Something has gone terribly wrong.")


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


    