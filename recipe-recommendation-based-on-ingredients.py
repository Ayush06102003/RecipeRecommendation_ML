#!/usr/bin/env python
# coding: utf-8

# In[1]:




# In[2]:


import numpy as np 
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pandas as pd
import pickle
import ast

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

warnings.simplefilter(action='ignore', category=Warning)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# This dataset consists of 230K recipes and 1.1M user-recipe interactions (reviews) scraped from Food.com (formerly GeniusKitchen), covering a period of 18 years (January 2000 to December 2018).

# In[3]:


raw_recipes=pd.read_csv('RAW_recipes.csv')
raw_recipes.head()
raw_recipes.columns
raw_recipes.shape


# In[4]:


raw_interactions=pd.read_csv('RAW_interactions.csv')
raw_interactions.head()
raw_interactions.columns
raw_interactions.shape


# In[5]:


ingr_map = joblib.load("ingr_map.pkl")
ingr_map.head()
ingr_map.columns
ingr_map.shape


# In[6]:


#rating numbers
raw_interactions.rating.value_counts(dropna=False)


# In[7]:


#rating = 0 
raw_interactions=raw_interactions[raw_interactions.rating>0]


# In[8]:


raw_interactions.recipe_id.nunique() #226590 unique recipes left


# In[9]:


#Group of recipes according to the number of comments and average
grouped=raw_interactions.groupby('recipe_id').agg({'rating':['count', 'mean']}).reset_index()
grouped.columns=grouped.columns.droplevel(0)
grouped.columns=['recipe_id','rating_count','rating_mean']
grouped.sort_values(by='rating_count',ascending=False)


# In[10]:




len(grouped[grouped['rating_count']>=40]) 
#The number of recipes with 40 or more comments is 2730
#the average rating of these 2730 recipes is between 3.4 and 5.
grouped[grouped['rating_count']>=40].rating_mean.max(), 
grouped[grouped['rating_count']>=40].rating_mean.min()
grouped[grouped['rating_count']>=40].rating_mean.value_counts() 


# In[11]:


#The IDs of 2730 recipes with more than 40 comments are added to a list.
recipes_to_filter=list(grouped[grouped['rating_count']>=40].recipe_id)


# In[12]:


#The recipe dataset is filtered with these 2730 recipes.
filtered_recipes=raw_recipes[raw_recipes.id.isin(recipes_to_filter)]
filtered_recipes.head(13)


# In[13]:


del raw_recipes
del raw_interactions


# 
# **Filtered Recipes Review**

# In[14]:


#Is there any duplication in the IDs or names of the recipes? 
filtered_recipes.id.nunique() #2730 There is no multiplexing in the ids 
filtered_recipes.name.nunique()#2728 , the names of the 2 recipes appear to be the same 
filtered_recipes.name.value_counts() #Recipes with the same name in the data twice: roasted brussels sprouts, strawberry rhubarb pie


# In[15]:



#Let's look at the details of the 2 recipes with multiple names.
filtered_recipes[filtered_recipes.name.isin(['roasted brussels sprouts','strawberry rhubarb pie'])]


# In[16]:




#It will not be possible to decide which of the 2 recipes with multiple names will be chosen, they will be eliminated. 
filtered_recipes=filtered_recipes[~filtered_recipes.name.isin(['roasted brussels sprouts','strawberry rhubarb pie'])] 
len(filtered_recipes) #2726 different recipes left


# 
# 
# # Nutrition

# In[17]:


filtered_recipes[['calories','total fat (PDV)','sugar (PDV)','sodium (PDV)','protein (PDV)','saturated fat (PDV)','carbohydrates (PDV)']] = filtered_recipes.nutrition.apply(lambda x: x.replace('[', '').replace(']', '')).str.split(",",expand=True).astype('float')
filtered_recipes.head(5)


# 
# 
# # Outlier Analyse

# In[18]:


#Outlier Functions
def outlier_thresholds(dataframe, col_name, q1, q3):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name, q1, q3):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1,q3)
    return dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None)
    
def replace_with_thresholds(dataframe, variable,q1,q3):
    low_limit, up_limit = outlier_thresholds(dataframe, variable,q1,q3)
    #dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


# In[19]:


filtered_recipes.describe()


# In[20]:


filtered_recipes.describe(percentiles=[0.8,0.9,0.95,0.99])


# In[21]:


filtered_recipes.dtypes


# In[22]:




#Numeric columns are selected for #outlier suppression.
num_cols = ['minutes', 'n_steps', 'n_ingredients', 'calories', 'total fat (PDV)', 'sugar (PDV)', 'sodium (PDV)', 'protein (PDV)', 'saturated fat (PDV)', 'carbohydrates (PDV)']


# In[23]:


print('outlier analysis for q1=0.01, q3=0.99')
for col in num_cols:
    print(f'{col} has outlier {check_outlier(filtered_recipes, col,q1=0.01, q3=0.99)} and values {outlier_thresholds(filtered_recipes,col,q1=0.01, q3=0.99)}')


# In[24]:


print('q1=0.05, q3=0.95 for outlier analysis')
for col in num_cols:
    print(f'{col} has outlier {check_outlier(filtered_recipes, col,q1=0.05, q3=0.95)} and values {outlier_thresholds(filtered_recipes,col,q1=0.05, q3=0.95)}')


# In[25]:


print('q1=0.2, q3=0.8for outlier analysis')
for col in num_cols:
    print(f'{col} has outlier {check_outlier(filtered_recipes, col,q1=0.2, q3=0.8)} and values {outlier_thresholds(filtered_recipes,col,q1=0.2, q3=0.8)}')


# # It was decided to fill the Outlier values with the upper limits of 0.8.

# In[26]:


for col in num_cols:
    filtered_recipes.col= replace_with_thresholds(filtered_recipes, col,q1=0.2, q3=0.8)


# In[27]:


filtered_recipes.describe()


# ### Records with minutes = 0, number of steps n_steps = 1, number of ingredients n_ingredients = 2 are examined.

# In[28]:


filtered_recipes[filtered_recipes.minutes==0]#subtracted from data
filtered_recipes= filtered_recipes[~(filtered_recipes.minutes==0)] #~ used for subtracting removing


# In[29]:


filtered_recipes[filtered_recipes.n_steps==1] #veride tutulur


# In[30]:


#filtered_recipes[filtered_recipes.n_ingredients==2]  


# In[31]:


#filtered_recipes[filtered_recipes.calories<=5] 


#  ### Necessary eliminations in the recipe file have been made. Recipe ids are kept in a dataframe and imported into a csv file

# In[32]:


#The index of the dataset is reset to start from 0.
filtered_recipes.reset_index(drop=True, inplace=True)


# In[33]:


filtered_recipes[['id','name']].to_csv('final_repices.csv',index=False) #only id-name matching


# #### Since the tags column in the data will also be used in the corpus, the values ​​to be selected by the user are exported to a CSV file to be used in the streamlit application.

# In[34]:


tag_lists = filtered_recipes.tags.tolist()
merged_list = []
for tag_list in tag_lists:
    merged_list.extend(tag_list)
    
    
distinct_tags = list(set(merged_list))
len(distinct_tags)


# In[35]:


# Among the #414 tags, selection will be made among those with high frequencies, their frequencies will be checked.
distinct_tags_df = pd.DataFrame({'tags': distinct_tags})
tag_frequencies = filtered_recipes['tags'].explode().value_counts().reset_index()
tag_frequencies = distinct_tags_df.merge(tag_frequencies, left_on='tags', right_on='index', how='left')
tag_frequencies.drop('index',axis=1,inplace=True)
tag_frequencies.columns = ['tag', 'frequency']

tag_frequencies = tag_frequencies.sort_values(by='frequency', ascending=False).reset_index(drop=True)
tag_frequencies


# In[36]:


tag_frequencies.describe().T


# In[37]:


#Tags below the average frequency are not included in the list for the user to choose from
tag_frequencies=tag_frequencies[~(tag_frequencies.frequency<140)]
len(tag_frequencies)


# In[38]:


#The 85 most frequently used tags are transferred to a csv file.
tag_frequencies[['tag']].to_csv('final_tags.csv',index=False) 


# ### The mapped ingredients in the material file are filtered through the selected recipe ids, kept in a dataframe and imported into a csv file. This file will create the list of materials to be selected by the user on streamlit for user inputs.

# In[39]:


pp_recipes=pd.read_csv('PP_recipes.csv')
filtered_pp=pp_recipes[pp_recipes.id.isin(filtered_recipes.id)]

import ast
filtered_pp['ingredient_id_eval'] =filtered_pp.ingredient_ids.apply(ast.literal_eval)
filtered_pp['ingredient_id_eval']


# In[40]:


pp_recipes=pd.read_csv('PP_recipes.csv')
filtered_pp=pp_recipes[pp_recipes.id.isin(filtered_recipes.id)]

import ast
filtered_pp['ingredient_id_eval'] =filtered_pp.ingredient_ids.apply(ast.literal_eval)

merged_ing_ids = set([value for sublist in filtered_pp.ingredient_id_eval for value in sublist])
len(merged_ing_ids)

ingr_map_filtered=ingr_map[ingr_map.id.isin(merged_ing_ids)]

final_ingr_list=list(set(ingr_map_filtered.replaced))

final_ingr_list=pd.DataFrame(final_ingr_list)

final_ingr_list.columns=['INGREDIENT']
final_ingr_list.to_csv('final_ingredients.csv',index=False)


# In[41]:


filtered_recipes.describe() 


# In[42]:


get_ipython().run_line_magic('matplotlib', 'inline')
fig, ax = plt.subplots()
filtered_recipes[num_cols].hist(ax=ax, bins=30)
fig.tight_layout()
plt.show()


# #### Cooking time (40 dk alti-Less than   40 mins & above- More than 40 mins)
# #### Calorie Restriction (400 cal alti-   Low Calory & above - High Calory)

# ### Final segment decision Nihai segment karari :
# 
# SEG1 - Less than 40 minutes & Low Calory
# 
# SEG2 - Less than 40 minutes & High Calory
# 
# SEG3 - More than 40 minutes & Low Calory
# 
# SEG4 - More than 40 minutes & High Calory

# In[43]:



print('SEG1 - Less than 40 minutes & Low Calory :',len(filtered_recipes[(filtered_recipes.minutes<=40)&(filtered_recipes.calories<=400)]))
print('SEG2 - Less than 40 minutes & High Calory :',len(filtered_recipes[(filtered_recipes.minutes<=40)&(filtered_recipes.calories>400)]))
print('SEG3 - More than 40 minutes & Low Calory :',len(filtered_recipes[(filtered_recipes.minutes>40)&(filtered_recipes.calories<=400)]))
print('SEG4 - More than 40 minutes & High Calory :',len(filtered_recipes[(filtered_recipes.minutes>40)&(filtered_recipes.calories>400)]))


# In[44]:


#SEG 1 
filtered_recipes[(filtered_recipes.minutes<=40)&(filtered_recipes.calories<=400)].describe()


# In[45]:


#SEG 2 
filtered_recipes[(filtered_recipes.minutes<=40)&(filtered_recipes.calories>400)].describe()


# In[46]:


#SEG 3 
filtered_recipes[(filtered_recipes.minutes>40)&(filtered_recipes.calories<=400)].describe()


# In[47]:


#SEG 4
filtered_recipes[(filtered_recipes.minutes>40)&(filtered_recipes.calories>400)].describe()


# In[48]:


#Segment assignments will be made after mapping procedures.


# # Ingredients Mapping

# In[49]:


filtered_recipes.head(3)


# #### Mapping the values in the ingredients list column with the 'replaced' column in the 'ingr_map' dataset and adding them to the dataset as a column named ingredients_mapped
# 

# In[50]:


ingr_map.head(3)


# In[51]:


import ast

# Create a dictionary mapping original ingredient names to mapped names
ingredient_mapping = dict(zip(ingr_map['raw_ingr'], ingr_map['replaced']))

# Function to replace ingredient names in a list
def replace_ingredients(ingredient_list):
    ingredients = ast.literal_eval(ingredient_list)
    return [ingredient_mapping.get(ingredient, ingredient) for ingredient in ingredients]

# Apply the function to the 'ingredients' column in filtered_recipes data frame
filtered_recipes['ingredients_mapped'] = filtered_recipes['ingredients'].apply(replace_ingredients)


# In[52]:


filtered_recipes.head(2)


# In[53]:


# Check one of the mappings, Filter the rows containing 'extra virgin olive oil'
filtered_rows = filtered_recipes[filtered_recipes['ingredients'].apply(lambda x: 'extra virgin olive oil' in x)]
filtered_rows.head(1)


# #### Merging the mapped BOM and tags into a list and adding it to the dataset as a column named merged_tags_ingredients
# 

# In[54]:


# Check one of the mappings, Filter the rows containing 'extra virgin olive oil'
x = filtered_rows.head(1)['ingredients']
for y in x:
    print(y)


# In[55]:


filtered_recipes['tags'] = filtered_recipes['tags'].apply(lambda x: eval(x) if isinstance(x, str) else x)


filtered_recipes['merged_tags_ingredients'] = filtered_recipes.apply(lambda row: row['ingredients_mapped'] + row['tags'], axis=1)
filtered_recipes['merged_tags_ingredients'] = filtered_recipes['merged_tags_ingredients'].apply(lambda lst: ', '.join(lst))

filtered_recipes.head(2)


# #### Nihai recipe dataframí bir csv ye export edilir.

# In[56]:


filtered_recipes.to_csv('final_repices_all.csv',index=False) #final 


#  ### Suggestions will be made based on the similarity of the words in the 'merged_tags_ingredients' column, where the tags and the mapped ingredient list are combined. The data is now ready to be divided into segments and similarity calculated through these segments.

# # Dividing the dataset into separate dataframes according to segments

# In[57]:


#Let's remember the segment structure and number again

print('SEG1 - Less than 40 minutes & Low Calory :',len(filtered_recipes[(filtered_recipes.minutes<=40)&(filtered_recipes.calories<=400)]))
print('SEG2 - Less than 40 minutes & High Calory :',len(filtered_recipes[(filtered_recipes.minutes<=40)&(filtered_recipes.calories>400)]))
print('SEG3 - More than 40 minutes & Low Calory :',len(filtered_recipes[(filtered_recipes.minutes>40)&(filtered_recipes.calories<=400)]))
print('SEG4 - More than 40 minutes & High Calory :',len(filtered_recipes[(filtered_recipes.minutes>40)&(filtered_recipes.calories>400)]))


# In[58]:


#Segment dataframes are created
seg1=filtered_recipes[(filtered_recipes.minutes<=40)&(filtered_recipes.calories<=400)]
seg2=filtered_recipes[(filtered_recipes.minutes<=40)&(filtered_recipes.calories>400)]
seg3=filtered_recipes[(filtered_recipes.minutes>40)&(filtered_recipes.calories<=400)]
seg4=filtered_recipes[(filtered_recipes.minutes>40)&(filtered_recipes.calories>400)]


# In[59]:


# Set the index of seg dataframes as the recipe names
seg1.set_index('name', inplace=True)
seg2.set_index('name', inplace=True)
seg3.set_index('name', inplace=True)
seg4.set_index('name', inplace=True)


# In[60]:


index_seg4 = seg4['steps']
index_seg4


# In[61]:


#del filtered_recipes


# # TF - IDF Vectorizer creation and distance evaluation via cosine similarity are turned into a function

# In[62]:


def tf_idf_vectorizer(dataframe,corpus_col):
    # Create the TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words="english")

    # Fit and transform the recipe corpus
    recipe_corpus = dataframe[corpus_col]
    tfidf_matrix = vectorizer.fit_transform(recipe_corpus)
    
    return vectorizer, tfidf_matrix


def get_similar_top5(vectorizer,tfidf_matrix,user_input):
    # Transform the user input using the fitted vectorizer
    user_input_tfidf = vectorizer.transform(user_input)
    
    # Compute the cosine similarity between user input and recipe corpus
    cosine_sim = cosine_similarity(user_input_tfidf, tfidf_matrix)
    
    # Get the indices of top 5 similar recipes
    similar_top5 = list(cosine_sim.argsort()[0][-5:])
    
    return similar_top5
    
    
def get_details_top5(dataframe, similar_top5):
    # Get the details of top 5 recommended recipes
    recommended_details = dataframe.iloc[similar_top5]
    return recommended_details[['ingredients_mapped', 'steps', 'minutes']]


# In[63]:


#The user_input list to be tested for prediction operations is created

#user_input = ['chicken', 'garlic', 'tomatoes', 'olive oil']
user_input = [ 'sugar','cream','bread','chocolate','strawberry']


# ## Establishing the model and making recommendations for SEG1

# In[64]:


#Model
seg1_vectorizer, seg1_tfidf_matrix = tf_idf_vectorizer(seg1,'merged_tags_ingredients')


similar_top5_indices=get_similar_top5(seg1_vectorizer,seg1_tfidf_matrix,user_input)
get_details_top5(seg1,similar_top5_indices)


# In[65]:


#seg1
pickle.dump(seg1_vectorizer, open('seg1_vectorizer.pkl','wb')) 
pickle.dump(seg1_tfidf_matrix, open('seg1_tfidf_matrix.pkl', 'wb'))


# 
# ## Establishing the model for SE and making recommendations & dumping pickle files

# In[66]:





#Model
seg2_vectorizer, seg2_tfidf_matrix = tf_idf_vectorizer(seg2,'merged_tags_ingredients')

#Recommending the top 5 recipes based on the user input entered
similar_top5_indices=get_similar_top5(seg2_vectorizer,seg2_tfidf_matrix,user_input)
get_details_top5(seg2,similar_top5_indices)


# In[67]:


#seg2
pickle.dump(seg2_vectorizer, open('seg2_vectorizer.pkl','wb')) 
pickle.dump(seg2_tfidf_matrix, open('seg2_tfidf_matrix.pkl', 'wb'))


# In[68]:


#Model 
seg3_vectorizer, seg3_tfidf_matrix = tf_idf_vectorizer(seg3,'merged_tags_ingredients')

#Recommending the top 5 recipes based on the user input entered
similar_top5_indices=get_similar_top5(seg3_vectorizer,seg3_tfidf_matrix,user_input)
get_details_top5(seg3,similar_top5_indices)


# In[69]:


#seg3
pickle.dump(seg3_vectorizer, open('seg3_vectorizer.pkl','wb')) 
pickle.dump(seg3_tfidf_matrix, open('seg3_tfidf_matrix.pkl', 'wb'))


# In[70]:


#Model 
seg4_vectorizer, seg4_tfidf_matrix = tf_idf_vectorizer(seg4,'merged_tags_ingredients')

#Recommending the top 5 recipes based on the user input entered
similar_top5_indices = get_similar_top5(seg4_vectorizer,seg4_tfidf_matrix,user_input)
get_details_top5(seg4,similar_top5_indices)


# In[71]:


#seg4
pickle.dump(seg4_vectorizer, open('seg4_vectorizer.pkl','wb')) 
pickle.dump(seg4_tfidf_matrix, open('seg4_tfidf_matrix.pkl', 'wb'))







# %%
