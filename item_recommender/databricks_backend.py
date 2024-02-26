# Databricks notebook source
# MAGIC %md
# MAGIC # Item Recommender Backend Code
# MAGIC
# MAGIC This is the (*as of 2/5/24*) current iteration of the Item Recommender on IRI/Circana data. See the "Item Attribution Project Testing" notebook for past iterations of this code.
# MAGIC
# MAGIC This code is designed to connect to a frontend web page hosted using Streamlit.

# COMMAND ----------

# Imports and API setup
import dbutils
import openai
import pandas as pd
import numpy as np
import requests
import json
import time

# Use newer API version here
# Connect to the Azure key vault for API secrets
# openai.api_key = dbutils.secrets.get(scope='adv-ds-secret', key='openai-west-key')
# openai.api_base = dbutils.secrets.get(scope='adv-ds-secret', key='openai-west-endpoint')
# openai.api_type = 'azure'
# openai.api_version = '2023-12-01-preview'

# Switch out deployments HERE
deployment_name='item-recommender-main' #This will correspond to the custom name you chose for your deployment when you deployed a model.

# COMMAND ----------
#
# iMPORT csv AND AVOID SPARK

df_awg =  pd.read_csv("C:/Users/norri/Desktop/awg_item_list.csv")

# Data cleaning and preprocessing
# df_awg =  read.table("awg_item_list")
#
# # Convert to Pandas dataframe
# df_awg = df_awg.toPandas()

# Create artifical null values - the number of nulls is higher here than in the UNFI dataset to stress test the model a bit
# Make sure the nulls only occur in the ADV columns though!
# Define the columns where you want to introduce NaNs and the source columns for the imputer to read 
target_columns = ['ADV_Brand', 'ADV_Category', 'ADV_SubCategory', 'ADV_ItemDescrip', 'ADV_ItemUPC', 'ADV_CaseUPC10', 'ADV_Size', 'ADV_StorePack'] # Need to make these read from the frontend in the future to be more programmatic
source_columns = ['Brand', 'Category Name', 'Sub Category Name', 'Item Description', 'UPCItem', 'UPCCase', 'Size', 'Store Pack'] # Also need to make sure the columns match up to the mapping set below

# Define the proportion of NaNs you want in your DataFrame  
nan_proportion = 0.2  # 20% NaNs  
  
# Calculate the number of values to replace with NaN for each column  
nan_count_per_column = {column: int(df_awg.shape[0] * nan_proportion) for column in target_columns}  
  
# Randomly choose indices to be replaced with NaN for each target column
np.random.seed(42) # Reproducibility
for column in target_columns:  
    nan_indices = np.random.choice(df_awg.index, nan_count_per_column[column], replace=False)  
    df_awg.loc[nan_indices, column] = np.nan  

# Separate this table into one free of null values and one containing nulls
df_awg_test = df_awg[~(df_awg.notna().all(axis=1))]

df_awg_train = df_awg[df_awg.notna().all(axis=1)]

# Make a dictionary to map the target column to the source data needed
# This allows for more dynamic mapping and multiple columns to inform a single target column's imputation
# To-do - determine this dict using the frontend directly instead of manually creating it
target_to_source_dfs = {target : pd.DataFrame(df_awg_train[column]) for (target, column) in zip(target_columns, source_columns)}

# COMMAND ----------

# MAGIC %md
# MAGIC Just take the artificially-nulled data from this cell (df_awg) and convert it into a csv for use in the backend function.

# COMMAND ----------

# Main Loop
# Put the finished data in an empty dataframe to avoid Pandas warnings 
imputed_target_df = pd.DataFrame()

# Loop through the dataframe column-wise and call the model to impute nulls in the ADV_ columns
for n, column in enumerate(target_columns):
  # Basic prompt setup - plug in column names from here from frontend
  # To-do - find a way to write this prompt dynamically based on the target_to_source_dfs dict or the frontend directly
  prompt = f"""
    The following data is a column of a dataset of product attributes called the AWG Item List: {df_awg_train[[source_columns[n]]].to_string(justify='center', sparsify=False, index=False)} 
    The missing item attribution values need to be filled in. Here's how missing values in the ADV_ column are imputed manually:
    The missing item attribution values need to be filled in. Here's how the columns map to each other:
    the ADV_Category column is based on the Category Name column,
    the ADV_Brand column is based on the Brand column,
    the ADV_SubCategory column is based on the Sub Category Name column,
    the ADV_ItemDescrip column is based on the Item Description column,
    the ADV_Size column is based on the Size column,
    the ADV_Store_Pack cloumn is based on the Store Pack column,
    the ADV_ItemUPC column is copied from the UPCItem column and should be formatted as a string (not in scientific notation),
    the ADV_CaseUPC10 column is exactly the same as the UPCCase column and should be formatted as a string (not in scientific notation),
    and the RptLvlFilter column values can be left null. 

    Autocomplete the data in the {column} column of the following validation sample from the AWG Item List based on the column mappings listed:
    {df_awg_test[[column]].to_string(justify='center', sparsify=False, index=False)} and format your answer into a series with no header or empty rows. The series will be appended to a new dataframe for export.
    You do not need to explain how to impute data or your methods for imputation - just output the completed data.
  """ 
  # While loop to prevent errors if the API doesn't connect successfully
  retries = 3
  data_out = None
  while retries > 0:    
    try: 
      print("Imputing...")
      # API call to OpenAI GPT-4 deployment
      response = openai.ChatCompletion.create(
        engine=deployment_name,
        temperature=1,
        messages=[
          {"role": "system", "content": "You are a model used to fill in missing data."},
          {"role": "user", "content": prompt}
          ]
      )
      data_out = response['choices'][0]['message']['content']
      print(f"Imputed in column {target_columns[n]}")
      time.sleep(2)
      break # End the while loop after it succeeds in calling the API. Might be bad practice to do things this way
    except Exception as e:
        print(e)  
        retries -= 1  
        if retries > 0:  
            print('Timeout error, retrying...')  
            time.sleep(5)  
    else:  
        print('API is not responding, all retries exhausted. Raising exception...')
        print('API is not responding with an accuracy above 50%. Attempting another run...')
        raise ValueError("Time out error - try restarting the script and checking your connection to OpenAI Studio")
  
  # Turn the output into a Pandas series
  # rows = pd.Series(data_out))
  # rows = data_out.split('\n')
  # rows = pd.Series(data_out.split('\n'))

  if data_out is not None:
      print('accuracy is below 50%')

  # data_out_series = pd.Series(rows[1:], name=column)

  # Append the series to the output df
  # imputed_target_df[column] = data_out_series


# COMMAND ----------

# Output
# Zip up the test data back with the training data now that its filled out
# Recombine the source column test data with the imputed data (filling in nulls). Might want to consider ditching the imputed_target_df var and appending straight to this though
df_awg_test.combine_first(imputed_target_df)

# Now combine this with the training data to recreate the original dataset but filled
df_output = pd.concat([df_awg_train, df_awg_test], axis=0)

# Read to csv for export
#df_output.to_csv('./export_test.csv', index=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Functionalizing the Backend Code

# COMMAND ----------

# Input csv will be df_awg after adding nulls
def backend_main(df_input):
    """
    Backend code for the Item Recommender placed into a single function for easy insertion into frontend code.
    input should be a dataframe version of the input csv.
    """
    # Convert the input to Pandas dataframe - for testing purposes, just input the df directly
    #df_input = pd.read_csv(input_csv)

    # Pre-processing - hard coding these values for now
    target_columns = ['ADV_Brand', 'ADV_Category', 'ADV_SubCategory', 'ADV_ItemDescrip', 'ADV_ItemUPC', 'ADV_CaseUPC10', 'ADV_Size', 'ADV_StorePack'] 
    source_columns = ['Brand', 'Category Name', 'Sub Category Name', 'Item Description', 'UPCItem', 'UPCCase', 'Size', 'Store Pack'] 
    # Separate this table into one free of null values and one containing nulls
    df_test = df_input[~(df_input.notna().all(axis=1))]
    df_train = df_input[df_input.notna().all(axis=1)]

    # Make a dictionary to map the target column to the source data needed
    target_to_source_dfs = {target : pd.DataFrame(df_train[column]) for (target, column) in zip(target_columns, source_columns)}

    # Main loop
    # Put the finished data in an empty dataframe to avoid Pandas warnings 
    imputed_target_df = pd.DataFrame()

    # Loop through the dataframe column-wise and call the model to impute nulls in the ADV_ columns
    for n, column in enumerate(target_columns):
        # Basic prompt setup - plug in column names from here from frontend
        # To-do - find a way to write this prompt dynamically based on the target_to_source_dfs dict or the frontend directly
        prompt = f"""
            The following data is a column of a dataset of product attributes called the AWG Item List: {df_train[[source_columns[n]]].to_string(justify='center', sparsify=False, index=False)} 
            The missing item attribution values need to be filled in. Here's how missing values in the ADV_ column are imputed manually:
            The missing item attribution values need to be filled in. Here's how the columns map to each other:
            the ADV_Category column is based on the Category Name column,
            the ADV_Brand column is based on the Brand column,
            the ADV_SubCategory column is based on the Sub Category Name column,
            the ADV_ItemDescrip column is based on the Item Description column,
            the ADV_Size column is based on the Size column,
            the ADV_Store_Pack cloumn is based on the Store Pack column,
            the ADV_ItemUPC column is copied from the UPCItem column and should be formatted as a string (not in scientific notation),
            the ADV_CaseUPC10 column is exactly the same as the UPCCase column and should be formatted as a string (not in scientific notation),
            and the RptLvlFilter column values can be left null. 

            Autocomplete the data in the {column} column of the following validation sample from the AWG Item List based on the column mappings listed:
            {df_test[[column]].to_string(justify='center', sparsify=False, index=False)} and format your answer into a series with no header or empty rows. The series will be appended to a new dataframe for export.
            You do not need to explain how to impute data or your methods for imputation - just output the completed data.
        """ 
        # While loop to prevent errors if the API doesn't connect successfully
        retries = 3    
        while retries > 0:    
            try: 
                print("Imputing...")
                # API call to OpenAI GPT-4 deployment
                response = openai.ChatCompletion.create(
                    engine=deployment_name,
                    temperature=1,
                    messages=[
                        {"role": "system", "content": "You are a model used to fill in missing data."},
                        {"role": "user", "content": prompt}
                        ]
                )
                data_out = response['choices'][0]['message']['content']
                print(f"Imputed in column {target_columns[n]}")
                time.sleep(2)
                break # End the while loop after it succeeds in calling the API. Might be bad practice to do things this way
            except Exception as e:
                print(e)  
                retries -= 1  
                if retries > 0:  
                    print('Timeout error, retrying...')  
                    time.sleep(5)  
            else:  
                print('API is not responding, all retries exhausted. Raising exception...')
                raise ValueError("Time out error - try restarting the script and checking your connection to OpenAI Studio")
  
        # Turn the output into a Pandas series
        rows = data_out.split('\n')
        data_out_series = pd.Series(rows[1:], name=column)

        # Append the series to the output df
        imputed_target_df[column] = data_out_series

    # Recombine the source column test data with the imputed data (filling in nulls)
    df_test.combine_first(imputed_target_df)

    # Now combine this with the training data to recreate the original dataset but filled
    df_output = pd.concat([df_train, df_test], axis=0)
    
    return df_output

# COMMAND ----------

# Test out the backend function
# df_output = backend_main(df_input=df_awg)
df_output = df_awg_test

print('Conclusion: the backend is not yet more accurate than a coin flip.)')
