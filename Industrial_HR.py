# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
import streamlit as st
import plotly.express as px


# Step 1: Merge all CSV data files and create a DataFrame
# Assuming all CSV files have similar structure, adjust as needed
file_paths = [r"C:\Users\jeeva\jika\Industrial Human Resource Geo-Visualization myself\drive\DDW_B18_0800_NIC_FINAL_STATE_RAJASTHAN-2011.csv", r"C:\Users\jeeva\jika\Industrial Human Resource Geo-Visualization myself\drive\DDW_B18_1200_NIC_FINAL_STATE_ARUNACHAL_PRADESH-2011.csv", r"C:\Users\jeeva\jika\Industrial Human Resource Geo-Visualization myself\drive\DDW_B18_1400_NIC_FINAL_STATE_MANIPUR-2011.csv", r"C:\Users\jeeva\jika\Industrial Human Resource Geo-Visualization myself\drive\DDW_B18_1500_NIC_FINAL_STATE_MIZORAM-2011.csv", r"C:\Users\jeeva\jika\Industrial Human Resource Geo-Visualization myself\drive\DDW_B18_1900_NIC_FINAL_STATE_WEST_BENGAL-2011.csv", r"C:\Users\jeeva\jika\Industrial Human Resource Geo-Visualization myself\drive\DDW_B18sc_0700_NIC_FINAL_STATE_NCT_OF_DELHI-2011.csv", r"C:\Users\jeeva\jika\Industrial Human Resource Geo-Visualization myself\drive\DDW_B18sc_1600_NIC_FINAL_STATE_TRIPURA-2011.csv", r"C:\Users\jeeva\jika\Industrial Human Resource Geo-Visualization myself\drive\DDW_B18sc_2000_NIC_FINAL_STATE_JHARKHAND-2011.csv", r"C:\Users\jeeva\jika\Industrial Human Resource Geo-Visualization myself\drive\DDW_B18sc_2400_NIC_FINAL_STATE_GUJARAT-2011.csv", r"C:\Users\jeeva\jika\Industrial Human Resource Geo-Visualization myself\drive\DDW_B18sc_2700_NIC_FINAL_STATE_MAHARASHTRA-2011.csv", r"C:\Users\jeeva\jika\Industrial Human Resource Geo-Visualization myself\drive\DDW_B18sc_2900_NIC_FINAL_STATE_KARNATAKA-2011.csv", r"C:\Users\jeeva\jika\Industrial Human Resource Geo-Visualization myself\drive\DDW_B18sc_3000_NIC_FINAL_STATE_GOA-2011.csv", r"C:\Users\jeeva\jika\Industrial Human Resource Geo-Visualization myself\drive\DDW_B18sc_3200_NIC_FINAL_STATE_KERALA-2011.csv", r"C:\Users\jeeva\jika\Industrial Human Resource Geo-Visualization myself\drive\DDW_B18sc_3300_NIC_FINAL_STATE_TAMIL_NADU-2011.csv", r"C:\Users\jeeva\jika\Industrial Human Resource Geo-Visualization myself\drive\DDW_B18sc_3400_NIC_FINAL_STATE_PUDUCHERRY-2011.csv", r"C:\Users\jeeva\jika\Industrial Human Resource Geo-Visualization myself\drive\DDW_B18st_0200_NIC_FINAL_STATE_HIMACHAL_PRADESH-2011.csv", r"C:\Users\jeeva\jika\Industrial Human Resource Geo-Visualization myself\drive\DDW_B18st_0500_NIC_FINAL_STATE_UTTARAKHAND-2011.csv", r"C:\Users\jeeva\jika\Industrial Human Resource Geo-Visualization myself\drive\DDW_B18st_0900_NIC_FINAL_STATE_UTTAR_PRADESH-2011.csv", r"C:\Users\jeeva\jika\Industrial Human Resource Geo-Visualization myself\drive\DDW_B18st_1000_NIC_FINAL_STATE_BIHAR-2011.csv", r"C:\Users\jeeva\jika\Industrial Human Resource Geo-Visualization myself\drive\DDW_B18st_1100_NIC_FINAL_STATE_SIKKIM-2011.csv", r"C:\Users\jeeva\jika\Industrial Human Resource Geo-Visualization myself\drive\DDW_B18st_1300_NIC_FINAL_STATE_NAGALAND-2011.csv", r"C:\Users\jeeva\jika\Industrial Human Resource Geo-Visualization myself\drive\DDW_B18st_1800_NIC_FINAL_STATE_ASSAM-2011.csv", r"C:\Users\jeeva\jika\Industrial Human Resource Geo-Visualization myself\drive\DDW_B18st_2100_NIC_FINAL_STATE_ODISHA-2011.csv"]
df_list = [pd.read_csv(file_path,encoding='latin1') for file_path in file_paths]
df = pd.concat(df_list, ignore_index=True)

# Step 2: Data Exploration, Cleaning, and Feature Engineering
# Perform necessary data exploration, cleaning, and feature engineering steps here

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['NIC Name'])
pca = PCA(n_components=2)
pca_result = pca.fit_transform(tfidf_matrix.toarray())
industry_embeddings = pca.fit_transform(tfidf_matrix.toarray())
df[['Industry_Embedding_1', 'Industry_Embedding_2']] = industry_embeddings
df['NIC Name'] = df['NIC Name'].apply(lambda x: float(x.replace('`', '')))

stop_words = set(stopwords.words('english'))
vectorizer = TfidfVectorizer(stop_words=stop_words)
subset_columns = ['State Code', 'District Code', 'India/States', 'Division', 'Group',
       'Class', 'NIC Name', 'Main Workers - Total -  Persons',
       'Main Workers - Total - Males', 'Main Workers - Total - Females',
       'Main Workers - Rural -  Persons', 'Main Workers - Rural - Males',
       'Main Workers - Rural - Females', 'Main Workers - Urban -  Persons',
       'Main Workers - Urban - Males', 'Main Workers - Urban - Females',
       'Marginal Workers - Total -  Persons',
       'Marginal Workers - Total - Males',
       'Marginal Workers - Total - Females',
       'Marginal Workers - Rural -  Persons',
       'Marginal Workers - Rural - Males',
       'Marginal Workers - Rural - Females',
       'Marginal Workers - Urban -  Persons',
       'Marginal Workers - Urban - Males',
       'Marginal Workers - Urban - Females']
X = df[subset_columns]
# Use clustering or classification algorithm to group industries, adjust as needed

# Step 4: Model Building and Testing
# Assuming 'target' is the column you want to predict
X_train, X_test, y_train, y_test = train_test_split(X, df['NIC Name'], test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Step 5: Streamlit Dashboard
st.title("Industrial HR Geo-Visualization Dashboard")

# Create plots using Plotly Express
fig1 = px.scatter_geo(data, lat="Latitude", lon="Longitude", color="Industry_Cluster",
                     hover_name="NIC Name", size="Main Workers - Total -  Persons",
                     projection="natural earth", title="Geographical Distribution of Industries")

fig2 = px.scatter(data, x="Industry_Embedding_1", y="Industry_Embedding_2",
                 color="Industry_Cluster", hover_name="NIC Name",
                 title="2D Embedding of Industries based on NIC Name")
st.plotly_chart(fig1)
st.plotly_chart(fig2)


