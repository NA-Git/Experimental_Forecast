import streamlit as st
import pandas as pd
import os
import shutil
import tempfile
#from openai import AzureOpenAI
#from azure.identity import DefaultAzureCredential, get_bearer_token_provider
# from openai import azureopenai
#import time
import itertools
import sys
from jarowinkler import jarowinkler_similarity
import numpy as np
from datawig import SimpleImputer

# Set the streamlit page to wide format for easier viewing
st.set_page_config(layout = "wide")

#token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")

# API info needed to call on OpenAI model
#client = AzureOpenAI(
    # https://learn.microsoft.com/en-us/azure/ai-services/openai/reference#rest-api-versioning
#    api_version='2023-12-01-preview',
    # https://learn.microsoft.com/en-us/azure/cognitive-services/openai/how-to/create-resource?pivots=web-portal#create-a-resource
#    azure_endpoint='',
   # azure_ad_token_provider=token_provider,
#    api_key= ''
#)

# Secondary backend function for additional imputation needs
# Datawig is used here
def backend_plus(df_input, issues, column_mapping):
    """
    Adds nulls where user-defined or ai detected issues are found
    Once these are added, they are imputed away using Datawig
    
    Parameters
    ----------
    df_input : Pandas dataframe. Data to be imputed in.
    issues : List of data issues found by the user or AI assistant.

    Returns
    -------
    Dataframe with no null values or data inconsistencies

    """
    # convert everything to strings to be safe
    df_input = df_input.astype('str') 
    
    # Create new nulls based on frontend input
    df_input.replace(issues, np.nan, inplace=True)
    
    # Run through datawig now that the first step is done - remove additional nulls
    SimpleImputer.complete(data_frame=df_input, inplace=True
                           ,precision_threshold=0.9
                           ,output_path="../datawig_temp/" # Specify where model data is stored to prevent errors
                          )
    # Return cleaned dataframe
    df_output = df_input.copy()
    return df_output

# Function for calling the backend code
# Input csv will be df_awg after adding nulls
def backend_main(df_input, column_mapping):
    """
    Backend code for the Item Recommender placed into a single function for easy insertion into frontend code.
    df_input should be the raw data directly from the user input on the frontend
    column_mapping is the user-inputted mappings converted into a Pandas dataframe
    -------
    Returns:
    dataframe with no null values
    """
    # Map the source columns to their intended target columns using frontend column mapping
    for ind in column_mapping.index:
        df_input[column_mapping['To'][ind]] = df_input[column_mapping['To'][ind]].fillna(df_input[column_mapping['From'][ind]])
    
    df_output = df_input.copy()
    
    return df_output

##### FRONT END CODE #####
#creates a session state for the column pairs
if 'columnsList' not in st.session_state:
    st.session_state["columnsList"] = []

#creates a session state for buttons, allows them to be grayed out so they can't be used once imputation has started
if "button1" not in st.session_state:
    st.session_state.button1 = False

if "button2" not in st.session_state:
    st.session_state.button2 = False

if "button3" not in st.session_state:
    st.session_state.button3 = False


# Flag to indicate if imputation has started
imputation_preview = False
imputation_preview_button_flag = False
imputation_started = False

#this callback unfreezes the Add Columns to SAved Column Pairs? button in the event that someone clicks the reset saved column pairs button
def reset_saved_columns_pair_callback():
    st.session_state.button2 = False #not st.session_state.button2

#this callback unfreezes all the buttons
def cancel_callback():
    st.session_state.button1 = not st.session_state.button1

#this callback allows the user to select yes in the imputation process
#def yes_callback():
 #   imputation_started = True


# Function to rename the file with "_new" added to the filename
def rename_file(filename):
    root, ext = os.path.splitext(filename)
    return root + '_new' + ext


# Frontend app implementation - functionalize the page setup
def frontend_main():
    """
    Function that contains most of the frontend code outside of some helper functions.

    Returns
    -------
    None. Only runs the code to set up the web page.
    """
    global imputation_started
    # Streamlit app with the new title
    st.title("Item Auto Attribution Tool")

    # File upload widget
    uploaded_file = st.file_uploader("**Upload a CSV file**", type=["csv"], disabled=st.session_state.button1)#, on_change=saved_columns.clear())

    # Initialize an empty dataframe to store saved columns
    saved_columns = []
    saved_template_columns = []

    finalDF = pd.DataFrame()
    # finalDF_Nulls = pd.DataFrame()

    #Once a file has been uploaded, triggers this portion of the code
    if uploaded_file is not None:
            df_input = pd.read_csv(uploaded_file, header=0)  # Try specifying header=0
            finalDF = df_input

            inputDF_Nulls = df_input.isna()
            finalDF_withFormatting = finalDF

            def color_null(val):
                color = '#d65f5f' if (pd.isnull(val)) else 'white'
                return f'background-color: {color}'

            with st.expander("**Imported Dataset**"):
                st.dataframe(finalDF_withFormatting.style.applymap(color_null), column_config={
                    "Item Code": st.column_config.TextColumn(),
                    "UPCItem": st.column_config.TextColumn(),
                    "UPCCase": st.column_config.TextColumn(),
                    "ADV_ItemUPC": st.column_config.TextColumn(),
                    "ADV_CaseUPC10": st.column_config.TextColumn(),
                },
                             hide_index=True,
                             )

            # Check if the DataFrame is not empty and contains columns
            if not df_input.empty:
                # Add a "Begin Imputation" button
                if st.button("**Begin Imputation**", key=1, disabled=st.session_state.button1): #, on_click=btn1_callback()):
                    imputation_started = True
                    st.session_state.button1 = True
                    st.session_state.button2 = True
                else:
                    st.session_state.button1 = False

                # Displays the column names with an expander button to collapse them
                columnsDisplayed = pd.DataFrame(df_input.columns.tolist())
                columnsDisplayed.columns = ["Column Names"]

                expander = st.sidebar.expander("**Column Names Detected**", expanded=False)
                expander.table(columnsDisplayed)

                # sc = pd.DataFrame()

                #section where the user can upload a template file of column mappings
                template_file = st.sidebar.file_uploader("**Upload a template file**", type=["csv"], disabled=st.session_state.button1)  # , on_change=saved_columns.clear())
                if (template_file is not None):
                    df_template_input = pd.read_csv(template_file,header=0)
                    st.sidebar.write(df_template_input)

                    #saved_columns.append(df_template_input)
                    if(st.sidebar.button("Add columns to Saved Column Pairs?", disabled=st.session_state.button2)):
                        #st.session_state.button2 = True
                        for i in df_template_input.index:
                            saved_template_columns.append([df_template_input.iloc[i,0], df_template_input.iloc[i,1]])
                            st.session_state["columnsList"].append([df_template_input.iloc[i,0], df_template_input.iloc[i,1]])
                        st.sidebar.success("**Selected columns saved!**")

                # Add two dropdown lists for column selection
                st.sidebar.header("Select Columns")
                column1 = st.sidebar.selectbox("**Select First Column**", df_input.columns.tolist(),  key="column1", disabled=st.session_state.button1)
                column2 = st.sidebar.selectbox("**Select Second Column**", df_input.columns.tolist(), key="column2", disabled=st.session_state.button1)

                # Add a button to save selected columns
                if st.sidebar.button("**Save Selected Columns**", key=3, disabled=st.session_state.button1):
                    if column1 == column2:
                        st.sidebar.warning("The columns selected must be different. Try again.")
                    else:
                    # Save the selected columns
                        saved_columns.append([column1, column2])
                        st.session_state["columnsList"].append([column1, column2])
                        st.sidebar.success("**Selected columns saved!**")

                #button to allow the user to reset the saved column pairs
                if st.sidebar.button('Reset Saved Column Pairs', key=4, disabled=st.session_state.button1, on_click = reset_saved_columns_pair_callback()):
                    saved_columns.clear()
                    #st.session_state.button2 = False
                    st.session_state["columnsList"].clear()

                #makes sure there are a unique set of column pairs for the user
                st.session_state["columnsList"].sort()
                st.session_state["columnsList"] = list(k for k,_ in itertools.groupby(st.session_state["columnsList"]))

                # Display saved columns at the bottom
                st.sidebar.subheader("**Saved Column Pairs**")
                savedColumnsDisplayed = pd.DataFrame(st.session_state["columnsList"])

                if (len(savedColumnsDisplayed) > 0):
                    savedColumnsDisplayed.columns = ['From', 'To']
                    st.sidebar.dataframe(savedColumnsDisplayed)

            else:
                st.warning("**The uploaded CSV file is empty or has no columns.**")


    # Check if imputation has started
    if imputation_started:
        # Perform imputation or any desired action here

        originalDS = df_input

        ##pulling in the columns pairs list from the front end section above
        # savedColumnMapping = pd.DataFrame(st.session_state["columnsList"])

        #cancels imputation and let's the user start over
        if(st.button('**Cancel**', on_click=cancel_callback())):
            st.session_state.button1 = False
            # kills the imputation process
            # However, you'll see the "Imputed in.." message pop up again once
            # This is because the API call already went out
            sys.exit('User canceled imputation. Resetting...')

        #Imputation process Kicks off
        # Prints a message letting the user know imputation has started
        with st.spinner("**Imputing...**"):
            # Column mapping imputation
            finalOutput = backend_main(df_input=finalDF, column_mapping=savedColumnsDisplayed)
            # Datawig imputation
            finalOutput = backend_plus(df_input=finalOutput, issues=['0'], column_mapping=savedColumnsDisplayed)
            #kick off the preview section
            #st.write("Here is a quick preview of what the results will look like when finished:")
            #st.write("Do you want to continue?")

        # kicks off the full imputation
        #if (st.button("Yes", on_click=yes_callback())):
         #   st.session_state.button1 = True

            #kicks off the full imputation
          #  with st.spinner("**Imputation process started....**"):
           #     time.sleep(5)
            st.success('Imputation Complete!')
        originalDS = originalDS.where(~inputDF_Nulls)

        # Create a temporary directory to save the renamed file
        temp_dir = tempfile.mkdtemp()
        new_filename = os.path.join(temp_dir, rename_file(uploaded_file.name))

        # Save the renamed file to the temporary directory
        finalOutput.to_csv(new_filename, index=False)
        
        finalOutput["Item Code"] = finalOutput["Item Code"].astype('int')

        finalOutputMerged= pd.merge(originalDS, finalOutput, how = "left", on=["Item Code"])

        st.success(f"**File saved as {new_filename}**")

        # Provide a download link for the new file
        st.download_button(
            label="**Download Processed File**",
            data=open(new_filename, "rb").read(),
            file_name=os.path.basename(new_filename),
            key="download_button",
        )

        ## Builds out the pandas data frame that identifies where the values were null and have been filled in
        inputDFNulls = inputDF_Nulls
        finalDFNulls = finalDF.isna()
        finalDFNullDiff = inputDFNulls.astype(int) - finalDFNulls.astype(int)
        finalOutputDisplayed = finalOutput

        # Function to apply the light green background for the missing values that were filled in
        def apply_styles(df,mask):
            # Create a styled DataFrame by copying the styles from the mask
            style = pd.DataFrame('', index=df.index, columns=df.columns)  # Initialize an empty style DataFrame
            for col in df.columns:
                style[col] = np.where(mask[col] == 1, 'background-color: lightblue', '')
            return style

        # Apply the style to the pandas dataframe
        finalOutputDisplayed['ADV_ItemUPC'] = finalOutputDisplayed['ADV_ItemUPC'].astype(str)
        styled_df = finalOutputDisplayed.style.apply(lambda x: apply_styles(finalOutputDisplayed, finalDFNullDiff), axis=None)


        ## This section outputs the final model data with the missing values filled in and colored green in the background to highlight where the data was filled in
        with st.expander("**Completed Model Data**"):
            st.dataframe(styled_df, column_config={
                "Item Code": st.column_config.TextColumn(),
                "UPCItem": st.column_config.TextColumn(),
                "UPCCase": st.column_config.TextColumn(),
                "ADV_ItemUPC": st.column_config.TextColumn(),
                "ADV_CaseUPC10": st.column_config.TextColumn(),
                                                 },
                     hide_index=True,
                     )

        def calculate_Jaro(x,y):
            return jarowinkler_similarity(str(x), str(y))

        finalOutputMerged['BrandJaroWinkler'] = finalOutputMerged.apply(lambda row: calculate_Jaro(row['Brand_x'], row['ADV_Brand_y']), axis = 1)
        finalOutputMerged['CategoryJaroWinkler'] = finalOutputMerged.apply(lambda row: calculate_Jaro(row['Category Name_x'], row['ADV_Category_y']), axis = 1)
        finalOutputMerged['SubCategoryJaroWinkler'] = finalOutputMerged.apply(lambda row: calculate_Jaro(row['Sub Category Name_x'], row['ADV_SubCategory_y']), axis=1)
        finalOutputMerged['ItemDescriptionJaroWinkler'] = finalOutputMerged.apply(lambda row: calculate_Jaro(row['Item Description_x'], row['ADV_ItemDescrip_y']), axis=1)
        finalOutputMerged['UPCItemJaroWinkler'] = finalOutputMerged.apply(lambda row: calculate_Jaro(row['UPCItem_x'], row['ADV_ItemUPC_y']), axis=1)
        finalOutputMerged['UPCCaseJaroWinkler'] = finalOutputMerged.apply(lambda row: calculate_Jaro(row['UPCCase_x'], row['ADV_CaseUPC10_y']), axis=1)
        finalOutputMerged['SizeJaroWinkler'] = finalOutputMerged.apply(lambda row: calculate_Jaro(row['Size_x'], row['ADV_Size_y']), axis=1)
        finalOutputMerged['StorePackJaroWinkler'] = finalOutputMerged.apply(lambda row: calculate_Jaro(row['Store Pack_x'], row['ADV_StorePack_y']), axis=1)

        ### This section is the model accuracy metrics
        ## First, calculate the accuracy metrics
        ##Old version commented out here
        #ADV_Brand_Accuracy = jarowinkler_similarity(finalOutputMerged['Brand_x'], finalOutputMerged['ADV_Brand_y'])
        #ADV_Category_Accuracy = jarowinkler_similarity(finalOutputMerged['Category Name_x'], finalOutputMerged['ADV_Category_y'])
        #ADV_SubCategory_Accuracy = jarowinkler_similarity(finalOutputMerged['Sub Category Name_x'], finalOutputMerged['ADV_SubCategory_y'])
        #ADV_ItemDescrip_Accuracy = jarowinkler_similarity(finalOutputMerged['Item Description_x'], finalOutputMerged['ADV_ItemDescrip_y'])
        #ADV_ItemUPC_Accuracy = jarowinkler_similarity(finalOutputMerged['UPCItem_x'], finalOutputMerged['ADV_ItemUPC_y'])
        #ADV_CaseUPC10_Accuracy = jarowinkler_similarity(finalOutputMerged['UPCCase_x'], finalOutputMerged['ADV_CaseUPC10_y'])
        #ADV_Size_Accuracy = jarowinkler_similarity(finalOutputMerged['Size_x'], finalOutputMerged['ADV_Size_y'])
        #ADV_StorePack_Accuracy = jarowinkler_similarity(finalOutputMerged['Store Pack_x'], finalOutputMerged['ADV_StorePack_y'])

        ADV_Brand_Accuracy = finalOutputMerged[finalOutputMerged['ADV_Brand_x'].isnull()]['BrandJaroWinkler'].mean()
        ADV_Category_Accuracy =  finalOutputMerged[finalOutputMerged['ADV_Category_x'].isnull()]['CategoryJaroWinkler'].mean()
        ADV_SubCategory_Accuracy =  finalOutputMerged[finalOutputMerged['ADV_SubCategory_x'].isnull()]['SubCategoryJaroWinkler'].mean()
        ADV_ItemDescrip_Accuracy =  finalOutputMerged[finalOutputMerged['ADV_ItemDescrip_x'].isnull()]['ItemDescriptionJaroWinkler'].mean()
        ADV_ItemUPC_Accuracy =  finalOutputMerged[finalOutputMerged['ADV_ItemUPC_x'].isnull()]['UPCItemJaroWinkler'].mean()
        ADV_CaseUPC10_Accuracy =  finalOutputMerged[finalOutputMerged['ADV_CaseUPC10_x'].isnull()]['UPCCaseJaroWinkler'].mean()
        ADV_Size_Accuracy =  finalOutputMerged[finalOutputMerged['ADV_Size_x'].isnull()]['SizeJaroWinkler'].mean()
        ADV_StorePack_Accuracy =  finalOutputMerged[finalOutputMerged['ADV_StorePack_x'].isnull()]['StorePackJaroWinkler'].mean()

        ## Second, display the accuracy metrics
        st.subheader("Model Metrics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("**ADV_Brand Column Accuracy %**", "{:.0%}".format(ADV_Brand_Accuracy))
        col2.metric("**ADV_Category Column Accuracy %**", "{:.0%}".format(ADV_Category_Accuracy))
        col3.metric("**ADV_SubCategory Column Accuracy %**", "{:.0%}".format(ADV_SubCategory_Accuracy))
        col4.metric("**ADV_ItemDescrip Column Accuracy %**", "{:.0%}".format(ADV_ItemDescrip_Accuracy))
        col5, col6, col7, col8 = st.columns(4)
        col5.metric("**ADV_ItemUPC Column Accuracy %**", "{:.0%}".format(ADV_ItemUPC_Accuracy))
        col6.metric("**ADV_CaseUPC10 Column Accuracy %**", "{:.0%}".format(ADV_CaseUPC10_Accuracy))
        col7.metric("**ADV_Size Column Accuracy %**", "{:.0%}".format(ADV_Size_Accuracy))
        col8.metric("**ADV_StorePack Column Accuracy %**", "{:.0%}".format(ADV_StorePack_Accuracy))

        # Clean up the temporary directory when the app is done
        shutil.rmtree(temp_dir)

        # Save the renamed file to the temporary directory
        #finaldf.to_csv(finalFileName, index=False)

    else:
        print('waiting')
    return # Don't need to return anything here, just run the script to create the webpage

# Run the frontend script via the above function
frontend_main()