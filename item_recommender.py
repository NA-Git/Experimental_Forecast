import streamlit as st
import pandas as pd
import os
import shutil
import tempfile
from openai import AzureOpenAI
from io import StringIO
#from azure.identity import DefaultAzureCredential, get_bearer_token_provider
# from openai import azureopenai
#import time
import itertools
import sys
from jarowinkler import jarowinkler_similarity
import numpy as np
from datawig import SimpleImputer
from mxnet.base import MXNetError

# Set the streamlit page to wide format for easier viewing
st.set_page_config(layout = "wide")

#token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")

# API info needed to call on OpenAI model
deployment_name='gpt-4o-2'

# API info needed to call on OpenAI model
client = AzureOpenAI(
    api_version='2024-02-01',
    azure_endpoint='https://adv-datascience-west.openai.azure.com/',
    api_key= 'd56952cf138e4c09a5f113682ce1b540'
)

def column_Mapping_backend(column_mapping):
    data_out = pd.DataFrame()
    retries = 3
    while retries > 0:
        try:
            column_mapping_copy = column_mapping.copy()
            column_mapping_copy_str = column_mapping_copy.astype('str')
            prompt = (f"The following dataset contains a list of column names:\n\n{column_mapping_copy_str}  "
                      f"Some column names with the ADV_ prefix are matching columns to the columns that do not contain the ADV_ prefix.  "
                      f"Can you attempt to pair up the ADV_ prefixed columns with the non ADV_ columns?  Please format your response into tabular dataset that will be easy to read back in python."
                      f"Please only respond with the completed table, with no other commentary. ")

            response = client.chat.completions.create(
                model = 'gpt-4o-2',
                #engine = deployment_name,
                max_tokens=1500,
                temperature = 1,
                messages= [
                    {"role": "system", "content": "You are a model that is used to assist in analyzing datasets."},
                    {"role": "user", "content": prompt}
                ]
            )
            data_out = response.choices[0].message.content
            break
        except Exception as e:
            print(e)
            retries -= 1
            if retries > 0:
                print('Timeout error, retrying...')
                #time.sleep(5)
        else:
          print('API is not responding, all retries exhausted. Raising exception...')
          raise ValueError("Time out error - try restarting the script and checking your connection to OpenAI Studio")

    #rows = data_out.split('\n')
    #data_out_series = pd.Series(rows[1:])

    # Cleaning the string to make it a valid CSV
    data = data_out.replace("|", "").strip()
    data = "\n".join([line.strip() for line in data.split("\n") if "---" not in line])

    # Use StringIO to read the string as a CSV
    df = pd.read_csv(StringIO(data), sep="\s\s+", engine='python')

    # Convert the list of lists into a Pandas DataFrame
    # data_out_series = pd.DataFrame(data[1:], columns=data[0])

    #data_out_series = pd.Series(data_out)
    df_output = df

    return df_output


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

if 'columnMapping_AI' not in st.session_state:
    st.session_state.columnMapping_AI = True

if 'columnsList_AI' not in st.session_state:
    st.session_state["columnsList_AI"] = []

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
    #global column_mapping_OpenAI_Flag
    # Streamlit app with the new title
    st.title("Item Auto Attribution Tool")

    # File upload widget
    uploaded_file = st.file_uploader("**Upload a CSV file**", type=["csv"],
                                     disabled=st.session_state.button1)  # , on_change=saved_columns.clear())

    # Initialize an empty dataframe to store saved columns
    saved_columns = []
    saved_template_columns = []
    columnsDisplayed = []
    savedColumnsDisplayed = []
    edited_savedColumnsDisplayed = []
    inputDF_Nulls = []
    columnsDisplayed_openAI = []

    finalDF = pd.DataFrame()
    #df_input = pd.read_csv(uploaded_file, header=0)
    # finalDF_Nulls = pd.DataFrame()



    # Once a file has been uploaded, triggers this portion of the code
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

        #st.session_state["columnsList"].clear()
        columnsDisplayed = pd.DataFrame(df_input.columns.tolist())
        # if (column_mapping_OpenAI_Flag):

        expander = st.sidebar.expander("**Column Names Detected**", expanded=False)
        expander.table(columnsDisplayed)


        if(len(st.session_state["columnsList_AI"])==0):
            st.session_state["columnMapping_AI"] = True

        if (st.session_state["columnMapping_AI"]):
            #st.session_state["columnsList_AI"] = column_Mapping_backend(columnsDisplayed)
            columnsDisplayed_openAI = column_Mapping_backend(columnsDisplayed)
            st.session_state["columnsList_AI"] = columnsDisplayed_openAI
            st.session_state["columnMapping_AI"] = False

        if (st.session_state["columnMapping_AI"] is not None):
            st.sidebar.write("**Below are the Column Mappings that were Auto Generated.**")
            st.sidebar.write(st.session_state["columnsList_AI"])

            # saved_columns.append(df_template_input)
            if (st.sidebar.button("**Add these Pairs to the Saved Column Pairs?**", disabled=st.session_state.button2)):
                st.sidebar.write("If any are not correct, you can remove them below.")
                # st.session_state.button2 = True
                for i in st.session_state["columnsList_AI"].index:
                    saved_template_columns.append([st.session_state["columnsList_AI"].iloc[i, 0], st.session_state["columnsList_AI"].iloc[i, 1]])
                    st.session_state["columnsList"].append(
                        [st.session_state["columnsList_AI"].iloc[i, 0], st.session_state["columnsList_AI"].iloc[i, 1]])
                st.sidebar.success("**Selected columns saved!**")


        # Check if the DataFrame is not empty and contains columns
        if not df_input.empty:
            # Add a "Begin Imputation" button
            if st.button("**Begin Imputation**", key=1,
                         disabled=st.session_state.button1):  # , on_click=btn1_callback()):
                imputation_started = True
                st.session_state.button1 = True
                st.session_state.button2 = True
            else:
                st.session_state.button1 = False

            columnsDisplayed.columns = ["Column Names"]

            # section where the user can upload a template file of column mappings
            template_file = st.sidebar.file_uploader("**Upload a template file**", type=["csv"],
                                                     disabled=st.session_state.button1)  # , on_change=saved_columns.clear())
            if (template_file is not None):
                df_template_input = pd.read_csv(template_file, header=0)
                st.sidebar.write(df_template_input)

                # saved_columns.append(df_template_input)
                if (st.sidebar.button("**Add columns to Saved Column Pairs?**", disabled=st.session_state.button2)):
                    # st.session_state.button2 = True
                    for i in df_template_input.index:
                        saved_template_columns.append([df_template_input.iloc[i, 0], df_template_input.iloc[i, 1]])
                        st.session_state["columnsList"].append(
                            [df_template_input.iloc[i, 0], df_template_input.iloc[i, 1]])
                    st.sidebar.success("**Selected columns saved!**")

            # Add two dropdown lists for column selection
            st.sidebar.header("Select Columns")
            column1 = st.sidebar.selectbox("**Select First Column**", df_input.columns.tolist(), key="column1",
                                           disabled=st.session_state.button1)
            column2 = st.sidebar.selectbox("**Select Second Column**", df_input.columns.tolist(), key="column2",
                                           disabled=st.session_state.button1)

            # Add a button to save selected columns
            if st.sidebar.button("**Save Selected Columns**", key=3, disabled=st.session_state.button1):
                if column1 == column2:
                    st.sidebar.warning("The columns selected must be different. Try again.")
                else:
                    # Save the selected columns
                    saved_columns.append([column1, column2])
                    st.session_state["columnsList"].append([column1, column2])
                    st.sidebar.success("**Selected columns saved!**")

            # button to allow the user to reset the saved column pairs
            if st.sidebar.button('Reset Saved Column Pairs', key=4, disabled=st.session_state.button1,
                                 on_click=reset_saved_columns_pair_callback()):
                saved_columns.clear()
                # st.session_state.button2 = False
                st.session_state["columnsList"].clear()

            # makes sure there are a unique set of column pairs for the user
            st.session_state["columnsList"].sort()
            st.session_state["columnsList"] = list(k for k, _ in itertools.groupby(st.session_state["columnsList"]))

            # Display saved columns at the bottom
            st.sidebar.subheader("**Saved Column Pairs**")
            savedColumnsDisplayed = pd.DataFrame(st.session_state["columnsList"])

            if (len(savedColumnsDisplayed) > 0):
                savedColumnsDisplayed.columns = ['From', 'To']
                #st.sidebar.data_editor(savedColumnsDisplayed, num_rows= "dynamic")
                edited_savedColumnsDisplayed =st.sidebar.data_editor(savedColumnsDisplayed, num_rows= "dynamic")
                #st.sidebar.dataframe(edited_savedColumnsDisplayed)
                #st.sidebar.dataframe(savedColumnsDisplayed)

               # st.session_state["columnsList"] =

                # Provide a download link for the column mapping
                st.sidebar.download_button(
                    label="**Download Column Mapping Template**",
                    data=edited_savedColumnsDisplayed.to_csv(index=False),
                    file_name=f"column_mapping_template_{uploaded_file.name}",
                    key="download_button_cm",
                )

        else:
            st.warning("**The uploaded CSV file is empty or has no columns.**")

    # Check if imputation has started
    if imputation_started:
        # Perform imputation or any desired action here

        originalDS = df_input

        ##pulling in the columns pairs list from the front end section above
        savedColumnMapping = pd.DataFrame(st.session_state["columnsList"])
        st.dataframe(edited_savedColumnsDisplayed)

        # cancels imputation and let's the user start over
        if (st.button('**Cancel**', on_click=cancel_callback())):
            st.session_state.button1 = False
            # kills the imputation process
            # However, you'll see the "Imputed in.." message pop up again once
            # This is because the API call already went out
            sys.exit('User canceled imputation. Resetting...')

        # Imputation process Kicks off
        # Prints a message letting the user know imputation has started
        with st.spinner("**Imputing...**"):
            # Column mapping imputation
            finalOutput = backend_main(df_input=finalDF, column_mapping=savedColumnsDisplayed)
            # Datawig imputation
            # Add a try-except block here to increase code stability
            #try:
                #finalOutput = backend_plus(df_input=finalOutput, issues=['0'], column_mapping=savedColumnsDisplayed)
            #except (FileNotFoundError,
                   # MXNetError):  # Ignore the "directory not found" errors since they don't seem to impact model output
                #finalOutput = backend_plus(df_input=finalOutput, issues=['0'], column_mapping=savedColumnsDisplayed)

            # kick off the preview section
            # st.write("Here is a quick preview of what the results will look like when finished:")
            # st.write("Do you want to continue?")

            # kicks off the full imputation
            # if (st.button("Yes", on_click=yes_callback())):
            #   st.session_state.button1 = True

            # kicks off the full imputation
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

        finalOutputMerged = pd.merge(originalDS, finalOutput, how="left", on=["Item Code"])

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
        def apply_styles(df, mask):
            # Create a styled DataFrame by copying the styles from the mask
            style = pd.DataFrame('', index=df.index, columns=df.columns)  # Initialize an empty style DataFrame
            for col in df.columns:
                style[col] = np.where(mask[col] == 1, 'background-color: lightblue', '')
            return style

        # Apply the style to the pandas dataframe
        # This should be ok to leave in since most item attribution data will have this feature
        finalOutputDisplayed['ADV_ItemUPC'] = finalOutputDisplayed['ADV_ItemUPC'].astype(str)
        styled_df = finalOutputDisplayed.style.apply(lambda x: apply_styles(finalOutputDisplayed, finalDFNullDiff),
                                                     axis=None)

        ## This section outputs the final model data with the missing values filled in and colored green in the background to highlight where the data was filled in
        # Same here - but maybe add a try/except block in the future just in case
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

        ## Calculate the accuracy metrics dynamically using Jaro-Winkler text similarity
        def calculate_Jaro(x, y):
            return jarowinkler_similarity(str(x), str(y))

        finalAccuracyMetrics = pd.DataFrame()

        # These for loops do the actual calculation by comparing the score pre-imputation to post-imputation (x vs y)
        for i in range(len(savedColumnMapping)):
            finalAccuracyMetrics[savedColumnMapping.loc[i, 1] + " Column Accuracy"] = np.nan
            finalOutputMerged[savedColumnMapping.loc[i, 1] + "_JaroWinkler"] = finalOutputMerged.apply(
                lambda row: calculate_Jaro(row[savedColumnMapping.loc[i, 0] + '_x'],
                                           row[savedColumnMapping.loc[i, 1] + '_y']), axis=1)

        # Output the metrics in a dataframe to be displayed on the frontend
        for i in range(len(savedColumnMapping)):
            finalAccuracyMetrics.loc[0, savedColumnMapping.loc[i, 1] + " Column Accuracy"] = \
            finalOutputMerged[finalOutputMerged[savedColumnMapping.loc[i, 1] + '_x'].isnull()][
                savedColumnMapping.loc[i, 1] + '_JaroWinkler'].mean()

        # Old accuracy metric code - delete after V1
        # ADV_Brand_Accuracy = finalOutputMerged[finalOutputMerged['ADV_Brand_x'].isnull()]['ADV_Brand_JaroWinkler'].mean()
        # ADV_Category_Accuracy =  finalOutputMerged[finalOutputMerged['ADV_Category_x'].isnull()]['ADV_Category_JaroWinkler'].mean()
        # ADV_SubCategory_Accuracy =  finalOutputMerged[finalOutputMerged['ADV_SubCategory_x'].isnull()]['ADV_SubCategory_JaroWinkler'].mean()
        # ADV_ItemDescrip_Accuracy =  finalOutputMerged[finalOutputMerged['ADV_ItemDescrip_x'].isnull()]['ADV_ItemDescrip_JaroWinkler'].mean()
        # ADV_ItemUPC_Accuracy =  finalOutputMerged[finalOutputMerged['ADV_ItemUPC_x'].isnull()]['ADV_ItemUPC_JaroWinkler'].mean()
        # #ADV_CaseUPC10_Accuracy =  finalOutputMerged[finalOutputMerged['ADV_CaseUPC10_x'].isnull()]['ADV_CaseUPC10_JaroWinkler'].mean()
        # ADV_Size_Accuracy =  finalOutputMerged[finalOutputMerged['ADV_Size_x'].isnull()]['ADV_Size_JaroWinkler'].mean()
        # ADV_StorePack_Accuracy =  finalOutputMerged[finalOutputMerged['ADV_StorePack_x'].isnull()]['ADV_StorePack_JaroWinkler'].mean()

        ## Second, display the accuracy metrics
        # Set up the section header and the number of columns for the metrics to be outputted in
        st.subheader("Model Metrics")
        col_num = 0  # This is used to count iteration number later
        col1, col2, col3, col4 = st.columns(4)
        output_columns = [col1, col2, col3, col4]  # Needed an iterable list of these variables

        # st.write(finalAccuracyMetrics) # testing

        # Dynamically loop through each column that gets imputed in and use the col list above to output metrics
        # The n col variables can be reused
        for i, n in enumerate(finalAccuracyMetrics.columns):
            output_columns[col_num].metric(f"**{n} %**", "{:.0%}".format(finalAccuracyMetrics.loc[0, n]))
            # Check to see if i has reached 4 or a number divisible by 4
            # This means a new output row needs to be started on the frontend and the col variables can be overwritten
            if (i + 1) % 4 == 0:
                col_num = 0
                pass
            # Otherwise keep going through the loop
            else:
                col_num += 1

        # Old accuracy metric code - delete after V1
        # col1, col2, col3, col4 = st.columns(4)
        # col1.metric("**ADV_Brand Column Accuracy %**", "{:.0%}".format(ADV_Brand_Accuracy))
        # col2.metric("**ADV_Category Column Accuracy %**", "{:.0%}".format(ADV_Category_Accuracy))
        # col3.metric("**ADV_SubCategory Column Accuracy %**", "{:.0%}".format(ADV_SubCategory_Accuracy))
        # col4.metric("**ADV_ItemDescrip Column Accuracy %**", "{:.0%}".format(ADV_ItemDescrip_Accuracy))
        # col1, col2, col3 = st.columns(3)
        # col1.metric("**ADV_ItemUPC Column Accuracy %**", "{:.0%}".format(ADV_ItemUPC_Accuracy))
        # #col6.metric("**ADV_CaseUPC10 Column Accuracy %**", "{:.0%}".format(ADV_CaseUPC10_Accuracy))
        # col2.metric("**ADV_Size Column Accuracy %**", "{:.0%}".format(ADV_Size_Accuracy))
        # col3.metric("**ADV_StorePack Column Accuracy %**", "{:.0%}".format(ADV_StorePack_Accuracy))

        # Clean up the temporary directory when the app is done
        shutil.rmtree(temp_dir)

        # Save the renamed file to the temporary directory
        # finaldf.to_csv(finalFileName, index=False)

    else:
        print('waiting')
    return  # Don't need to return anything here, just run the script to create the webpage


# Run the frontend script via the above function
frontend_main()