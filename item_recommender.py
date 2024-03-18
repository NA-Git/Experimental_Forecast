import streamlit as st
import pandas as pd
import os
import shutil
import tempfile
from openai import AzureOpenAI
#from azure.identity import DefaultAzureCredential, get_bearer_token_provider
# from openai import azureopenai
import time
import itertools
import sys
import numpy as np

#token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")

    # API info needed to call on OpenAI model
#azureopenai.api_key = 'd56952cf138e4c09a5f113682ce1b540'
#azureopenai.api_base = 'https://adv-datascience-west.openai.azure.com/'
#azureopenai.api_type = 'azure'
#azureopenai.api_version = '2023-12-01-preview'
deployment_name='item-recommender-main'

client = AzureOpenAI(
    # https://learn.microsoft.com/en-us/azure/ai-services/openai/reference#rest-api-versioning
    api_version='2023-12-01-preview',
    # https://learn.microsoft.com/en-us/azure/cognitive-services/openai/how-to/create-resource?pivots=web-portal#create-a-resource
    azure_endpoint='https://adv-datascience-west.openai.azure.com/',
   # azure_ad_token_provider=token_provider,
    api_key= 'd56952cf138e4c09a5f113682ce1b540'
)


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
    # Convert the input to Pandas dataframe - for testing purposes, just input the df directly
    # df_input = pd.read_csv(input_csv)
    for ind in column_mapping.index:
        df_input[column_mapping['To'][ind]] = df_input[column_mapping['To'][ind]].fillna(df_input[column_mapping['From'][ind]])
        df_output = df_input.copy()
    
    return df_output


##### FRONT END CODE
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

    #Once a file has been uploaded, triggers this portion of the code
    if uploaded_file is not None:
        #try:
            # Attempt to read the CSV file with different settings
            df_input = pd.read_csv(uploaded_file, header=0)  # Try specifying header=0
            finalDF = df_input

            # Check if the DataFrame is not empty and contains columns
            if not df_input.empty:
                # Add a "Begin Imputation" button
                if st.button("**Begin Imputation**", key=1, disabled=st.session_state.button1): #, on_click=btn1_callback()):
                    imputation_started = True
                    st.session_state.button1 = True
                    st.session_state.button2 = True
                else:
                    st.session_state.button1 = False

                #if st.button("**Imputation Preview**", key=2, disabled=st.session_state.button1):
                #    imputation_preview = True
                 #   st.session_state.button1 = True
                #else:
                 #   st.session_state.button1 = False

                # Displays the column names with an expander button to collapse them
                columnsDisplayed = pd.DataFrame(df_input.columns.tolist())
                columnsDisplayed.columns = ["Column Names"]

                expander = st.sidebar.expander("**Column Names Detected**", expanded=False)
                expander.table(columnsDisplayed)

                sc = pd.DataFrame()

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

    #Check if the imputation preview has been started
    #if imputation_preview:
        # cancels imputation preview and let's the user start over
     #   if (st.button('**Cancel Preview**')):
          #  st.session_state.list.clear()

        #st.write("**Imputation preview started...**")
      #  with st.spinner("**Imputation preview started...**"):
       #     time.sleep(5)
        #    imputation_preview_button_flag = True
            #st.button('**Cancel Preview**', st.empty=True)
        #st.success('Done!')
        #if(imputation_preview_button_flag):
         #   st.session_state.list.clear()



    # Check if imputation has started
    if imputation_started:
        # Perform imputation or any desired action here
        # Prints a message letting the user know imputation has started
        #st.spinner("**Imputation preview started...**")
        #st.write("**Imputation process started...**")

        ##pulling in the columns pairs list from the front end section above
        savedColumnMapping = pd.DataFrame(st.session_state["columnsList"])

        #cancels imputation and let's the user start over
        if(st.button('**Cancel**', on_click=cancel_callback())):
            st.session_state.button1 = False
            # kills the imputation process
            # However, you'll see the "Imputed in.." message pop up again once
            # This is because the API call already went out
            sys.exit('User canceled imputation. Resetting...')

            #Imputation process Kicks off
        with st.spinner("**Imputation process started...**"):
            #time.sleep(5)  ##remove this once the preview piece is done
            finalOutput = backend_main(df_input=finalDF, column_mapping=savedColumnsDisplayed)
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
            #kills off the process

        #if(st.button("No")):
         #   st.session_state.list.clear()
          #  st.session_state.button1 = False
            #finalOutput = backend_main(finalDF)

        #finalOutput = finalDF


        # Create a temporary directory to save the renamed file
        temp_dir = tempfile.mkdtemp()
        new_filename = os.path.join(temp_dir, rename_file(uploaded_file.name))

        # Save the renamed file to the temporary directory
        finalOutput.to_csv(new_filename, index=False)

        st.success(f"**File saved as {new_filename}**")

        # Provide a download link for the new file
        st.download_button(
            label="**Download Processed File**",
            data=open(new_filename, "rb").read(),
            file_name=os.path.basename(new_filename),
            key="download_button",
        )

        # Clean up the temporary directory when the app is done
        shutil.rmtree(temp_dir)

        # Save the renamed file to the temporary directory
        #finaldf.to_csv(finalFileName, index=False)

    else:
        print('waiting')
    return # Don't need to return anything here, just run the script to create the webpage

# Run the frontend script via the above function
frontend_main()