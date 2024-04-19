import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt
import seaborn as sns

import streamlit as st

def upload_dataset():
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        return df
    else:
        return None

def manually_enter_data():
    st.write("## Enter Data Manually")

# Streamlit UI function to collect user inputs for country
def collect_country():
    country = st.selectbox("Country", ['Australia', 'Canada', 'Ireland', 'UK', 'USA'])
    return country

def format_political_features(country):
    # Fixed political features
    fixed_political_features = {
        "Australia": [44292.16420986, 0.94992879, 10.6, 150.24454998, 43.81482342],
        "Canada": [43891.22986816, 1.09891699, 13.31818182, 147.19275946, 25.39082926],
        "Ireland": [56635.51385961, 0.98566395, 9.73636364, 75.82689221, 24.43034975],
        "UK": [40433.93304994, 0.36291698, 11.88181818, 134.04335217, 75.64203555],
        "USA": [53821.95755477, 0.51039351, 17.86363636, 245.11261028, 34.16351205]
    }
   
    # Create DataFrame with the fixed political features for the selected country
    country_features = pd.DataFrame([fixed_political_features[country]], columns=["GDP", "Politics", "Poverty", 
                                                                                  "Serious assault", "Sexual violence"])
    
    encoded_country = pd.DataFrame(columns=['Australia', 'Canada', 'Ireland', 'UK', 'USA'])
    
    # Assign 1 to the selected country's column
    encoded_country[country] = 1
    
    # Assign 0 to all other country columns
    for c in ['Australia', 'Canada', 'Ireland', 'UK', 'USA']:
        if c != country:
            encoded_country[c] = 0
        
    # Concatenate the one-hot encoded country and political features with the existing DataFrame
    df = pd.concat([encoded_country, country_features], axis=1)
    
    return df

# Streamlit UI function to collect user inputs for age

def collect_age():
    age = st.selectbox("Age", ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"], key="age")
    return age

def format_age(df, age):
    mapping_age = {"18-24" : 0,
                   "25-34": 1,
                   "35-44": 2,
                   "45-54": 3,
                   "55-64": 4,
                   "65+": 5}

    # Map the selected age range to its corresponding numerical value
    age_value = mapping_age[age]

    # Populate the "Age" column with the mapped value
    df["Age"] = age_value
    
    return df

# Streamlit UI function to collect user inputs for gender

def collect_gender():
    gender = st.radio("Gender", ["Female", "Male"])
    return gender

def format_gender(df,gender):
    mapping_gender = {"Female" : 1,
                     "Male": -1}
    
    # Map the selected age range to its corresponding numerical value
    gender_value = mapping_gender[gender]

    # Populate the "Gender" column with the mapped value
    df["Gender"] = gender_value
    
    return df
# Streamlit UI function to collect user inputs for education

def collect_education():
    education = st.selectbox("Education", ["Left School at 18 years", "High School Diploma", 
                                           "University Degree", "Masters Degree", "Doctorate Degree"], key="education")
    return education

def format_edu(df, education):
    mapping_edu = {"Left School at 18 years": 0,
                   "High School Diploma": 1,
                   "University Degree": 2,
                   "Masters Degree": 2,
                   "Doctorate Degree": 2}
    
    # Map the selected age range to its corresponding numerical value
    edu_value = mapping_edu[education]

    # Populate the "Gender" column with the mapped value
    df["Education"] = edu_value
    
    return df

# Streamlit UI function to calculate the right unemployment rate based on the education, gender and country entered by the user

def get_unemployment_rate(df, education, gender, country):

    unemployment_rates = {
        (0, 1, "Australia"): 9.373909,
        (0, 1, "Canada"): 13.938727,
        (0, 1, "Ireland"): 15.342636,
        (0, 1, "UK"): 9.912545,
        (0, 1, "USA"): 11.944182,
        (1, 1, "Australia"): 5.990455,
        (1, 1, "Canada"): 7.442909,
        (1, 1, "Ireland"): 11.881091,
        (1, 1, "UK"): 6.503273,
        (1, 1, "USA"): 8.231545,
        (2, 1, "Australia"): 3.288273,
        (2, 1, "Canada"): 5.076545,
        (2, 1, "Ireland"): 6.216636,
        (2, 1, "UK"): 3.315364,
        (2, 1, "USA"): 3.926545,
        (0, -1, "Australia"): 9.959909,
        (0, -1, "Canada"): 14.438273,
        (0, -1, "Ireland"): 18.537545,
        (0, -1, "UK"): 11.742455,
        (0, -1, "USA"): 9.562727,
        (1, -1, "Australia"): 9.959909,
        (1, -1, "Canada"): 8.676636,
        (1, -1, "Ireland"): 14.414000,
        (1, -1, "UK"): 6.948545,
        (1, -1, "USA"): 9.088545,
        (2, -1, "Australia"): 3.120545,
        (2, -1, "Canada"): 5.757364,
        (2, -1, "Ireland"): 6.141818,
        (2, -1, "UK"): 3.527909,
        (2, -1, "USA"): 3.852182}
    
    # Get the unemployment rate based on the provided parameters
    rate = unemployment_rates.get((education, gender, country), 0.0)
    
     # Create a new column in the DataFrame for unemployment rate
    df['Unemployment_Rate'] = rate
    
    return df

# Streamlit UI function to collect user inputs for each psychological trait
# values=x default average values

def collect_Nscore():
    return st.slider("Neuroticism score (Nscore)", min_value=12, max_value=60, value=36) 

def collect_Escore():
    return st.slider("Extraversion score (Escore)", min_value=16, max_value=59, value=37)

def collect_Oscore():
    return st.slider("Openness to experience score (Oscore)", min_value=24, max_value=60, value=42)

def collect_Ascore():
    return st.slider("Agreeableness score (Ascore)", min_value=24, max_value=60, value=40)

def collect_Cscore():
    return st.slider("Conscientiousness score (Cscore)", min_value=17, max_value=59, value=38)

def collect_Impulsive():
    return st.selectbox("Impulsiveness score (Impulsive)", options=[-2.55524, -1.37983, -0.71126, 
                                                                    -0.21712, 0.19268, 0.52975, 0.88113, 
                                                                    1.29221, 1.86203, 2.90161], index=4, key="imp")

def collect_SS():
    return st.selectbox("Sensation score (SS)", options=[-2.07848, -1.54858, -1.18084, 
                                                         -0.84637, -0.52593, -0.21575, 
                                                         0.07987, 0.40148, 0.76540, 1.22470,
                                                         1.92173], index=5, key="ss")

# Function to collect scores for all traits
def collect_user_traits():
    trait_scores = {
        "Neuroticism": collect_Nscore(),
        "Extraversion": collect_Escore(),
        "Openness to experience": collect_Oscore(),
        "Agreeableness": collect_Ascore(),
        "Conscientiousness": collect_Cscore(),
        "Impulsiveness": collect_Impulsive(),
        "Sensation": collect_SS()
    }
    return trait_scores

# Function to standardize the features and concatenate horizontally with existing df
def standardize_and_concatenate_features(trait_scores, df):
    
    # Convert trait scores dictionary to DataFrame
    trait_df = pd.DataFrame.from_dict(trait_scores, orient='index', columns=['Score'])
    
    # Standardize the trait features using StandardScaler
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(trait_df)
    
    # Convert standardized features back to DataFrame
    trait_df_scaled = pd.DataFrame(scaled_features, index=trait_df.index, columns=trait_df.columns)
    columns_standardized = ["GDP", "Politics", "Poverty", "Serious assault", "Sexual violence"]
    df[columns_standardized] = scaler.fit_transform(df[columns_standardized])
    
    # Concatenate horizontally with existing DataFrame
    concatenated_df = pd.concat([df, trait_df_scaled.T], axis=1)
    
    return concatenated_df


# Function to convert DataFrame to tensor
def df_to_tensor(concatenated_df):
    # Convert to numpy array
    np_array = concatenated_df.astype('float64').to_numpy()
    # Convert to tensor
    tensor = torch.FloatTensor(np_array)
    return tensor

# Function to make predictions using the model
def make_predictions(model, tensor):
    with torch.no_grad():
        predictions = model.forward(tensor)
    return predictions

class NeuralNet(nn.Module):
    
    def __init__(self, input_size=21, hidden_size1=120, hidden_size2=100, output_size=6):
        super(NeuralNet, self).__init__() # Instantiate out nn.Model 
        # Initiating the layers (fc = fully connected)
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.out = nn.Linear(hidden_size2, output_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):  # Define the function to move forward in the network
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.out(out)
        out = self.sigmoid(out)
    
        return out

torch.manual_seed(40)    
model = NeuralNet()

default_df = pd.read_csv("../data/df_tot.csv")

def explore_data(df):
    st.title("Explore Data")
    st.write("### Sample Data:")
    st.write(df.head())

    st.write("### Use of Subtances by Country")
    
    drugs_country = ['Country','Alcohol', 'CNS_depressants','CNS_stimulants', 'Cannabis', 'Nicotine', 'Hallucinogens']
    substances_df = df[drugs_country].set_index("Country")
    melted_df = substances_df.reset_index().melt(id_vars='Country', var_name='Substance', value_name='Usage')

    plt.figure(figsize=(10,5))
    sns.barplot(data=melted_df, x = "Country", y="Usage", hue="Substance", errorbar=("ci", 50), capsize=.4,
        err_kws={"color": ".5", "linewidth": 1.5},
        linewidth=1.5, edgecolor=".5", palette = "pastel")

    plt.ylim(0,6)
    plt.title("Use of substances by Country", fontsize=15)
    plt.xlabel("Country", fontsize=13)
    plt.ylabel("Level of Usage", fontsize=13)
    plt.legend(title='Substances', bbox_to_anchor=(1, 1.02), loc='upper left')
    sns.despine(top=True, right=True)
    st.pyplot()
    st.set_option('deprecation.showPyplotGlobalUse', False)

    st.write("#### Use of substances VS Unemployment")
    drugs_unempl = ['Unemployment_Rate','Alcohol', 'CNS_depressants','CNS_stimulants', 'Cannabis', 'Nicotine', 'Hallucinogens']
    substances_df_unempl = df[drugs_unempl].set_index("Unemployment_Rate")


    melted_df_unempl = substances_df_unempl.reset_index().melt(id_vars='Unemployment_Rate', 
                                                            var_name='Substance', value_name='Usage')


    sns.lineplot(data=melted_df_unempl, x="Usage", y="Unemployment_Rate", hue="Substance", markers=True, linewidth=2.5)

    plt.title("Use of substances VS Unemployment", fontsize=15)
    plt.xlabel("Level of Usage", fontsize=13)
    plt.ylabel("Unemployment Rate", fontsize=13)
    plt.legend(title='Substances', bbox_to_anchor=(1, 1.02), loc='upper left')
    sns.despine(top=True, right=True)
    plt.ylim(4.5,10)
    st.pyplot()
    st.set_option('deprecation.showPyplotGlobalUse', False)

    st.write("#### Use of substances VS GDP")
    drugs_gdp = ['GDP','Alcohol', 'CNS_depressants','CNS_stimulants', 'Cannabis', 'Nicotine', 'Hallucinogens']
    substances_df_gdp = df[drugs_gdp].set_index("GDP")

    melted_df_gdp = substances_df_gdp.reset_index().melt(id_vars='GDP', 
                                                            var_name='Substance', value_name='Usage')

    sns.lineplot(data=melted_df_gdp, x="Usage", y="GDP", hue="Substance", markers=True, linewidth=2.5)

    plt.title("Use of substances VS GDP", fontsize=15)
    plt.xlabel("Level of Usage", fontsize=13)
    plt.ylabel("GDP pro capite ($)", fontsize=13)
    plt.legend(title='Substances', bbox_to_anchor=(1, 1.02), loc='upper left')
    sns.despine(top=True, right=True)
    st.pyplot()
    st.set_option('deprecation.showPyplotGlobalUse', False)

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ("Options", "Explore"))

    if page == "Options":
        st.title("Options")
        data_option = st.radio("Select Data Input Option", ("Upload Dataset", "Enter Data Manually"))

        if data_option == "Upload Dataset":
            df = upload_dataset()
        else:
            df = manually_enter_data()

        # Collect user inputs for country
        country = collect_country()

        # Format political features based on country
        df = pd.DataFrame()  # Create an empty DataFrame
        df = format_political_features(country)  # Reassign the DataFrame after formatting political features

        # Collect user inputs for age
        age = collect_age()

        # Format age
        df = format_age(df, age)

        # Collect user inputs for gender
        gender = collect_gender()

        # Format gender
        df = format_gender(df, gender)

        # Collect user inputs for education
        education = collect_education()

        # Format education
        df = format_edu(df, education)

        # Collect user inputs for each psychological trait
        trait_scores = collect_user_traits()

        # Standardize and concatenate features
        concatenated_df = standardize_and_concatenate_features(trait_scores, df)

        # Get unemployment rate and update DataFrame
        updated_df = get_unemployment_rate(concatenated_df, education, gender, country)

        tensor = df_to_tensor(updated_df)

        # Make predictions using the model
        predictions = make_predictions(model, tensor)
        
        class_labels = {
            0: 'Alcohol',
            1: 'CNS depressants',
            2: 'CNS stimulants',
            3: 'Cannabis',
            4: 'Nicotine',
            5: 'Hallucinogens'
        }

        predicted_labels = []
        for i, prediction in enumerate(predictions):
            class_label = class_labels[i]
            predicted_class = "User" if torch.round(prediction).any().item() == 1 else "Non-User"
            predicted_labels.append((class_label, predicted_class))

        # Display the predictions for each class
        st.title("Predictions for Drug Use")
        for class_label, predicted_class in predicted_labels:
            st.write(f"{class_label}: {predicted_class}")

        if df is not None:
            st.write("### Preview of Data")
            st.write(df.head())
   
    elif page == "Explore":
        # Call the function to explore data here
        explore_data(default_df)  # Use the default DataFrame for exploration
    
if __name__ == "__main__":
    main()
    