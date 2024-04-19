import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def histo_numerical(df, column):
    """
    The function calculates the skewness and kurtosis of a specified column in a DataFrame. 
    It prints the values of the two characteristics and display the histogram plot of the 
    data distribution.

    Parameters:
        df (DataFrame): The DataFrame containing the data.
        column (str): The name (as string) of the column for which the distribution is shown.
    """

    column_data = df[column]

    skewness = column_data.skew()
    kurtosis = column_data.kurtosis()
    print(f"""Distribution measures:
    Skewness: {skewness}
    Kurtosis: {kurtosis}""")

    plt.figure(figsize = (3,3))
    sns.histplot(column_data, bins = 20, kde = True)


def prioritize_values(series):
    """
    The function prioritizes the values in a pandas Series based on a predefined priority list.
    It returns the highest priority value in the Series according to the predefined list.
    
    Parameters:
    - series: A pandas Series containing values to be prioritized.
    """
    
    sorted_values = sorted(series, key=lambda x: class_priority.index(x) if x in class_priority else float('inf'))
    
    return sorted_values[0]

def drugs_cat(df, column_name):

	"""
	Function to merge columns with similar names in a DataFrame, 
	prioritizing values based on the prioritize_values function.
	It returns the modified DataFrame with the merged column added and the original columns dropped.
	
	Parameters:
	- df: DataFrame (input DataFrame containing the columns to be categorized and merged).
	- column_name: string (the common label of the columns to be merged)
  	"""
    
    specific_column = [col for col in df.columns if col.startswith(column_name)]
    merged_column = df[specific_column].apply(prioritize_values, axis=1)
    df.drop(columns=specific_column, inplace=True)
    
    df[column_name] = merged_column
    
    return df

 def select_unemployment_rate(row):

 	"""
 	Function to select the correspondent unemployment rate for each individual 
 	based on their level of education and gender.
 	Parameter:
 	- row: each row of the DataFrame
 	To be applied with df.apply() for example to create another column havig as row the values returned by the function.
 	"""
    
    if row["Education"] == 0:
        if row["Gender"] == -1:  # male = -1
            return row["unemployment_bas_edu_m"]
        elif row["Gender"] == 1:  # female = 1
            return row["unemployment_bas_edu_f"]
    elif row["Education"] == 1:
        if row["Gender"] == -1:
            return row["unemployment_int_edu_m"]
        elif row["Gender"] == 1:
            return row["unemployment_int_edu_f"]
    elif row["Education"] == 2:
        if row["Gender"] == -1:
            return row["unemployment_adv_edu_m"]
        elif row["Gender"] == 1:
            return row["unemployment_adv_edu_f"]
    else:
        return None