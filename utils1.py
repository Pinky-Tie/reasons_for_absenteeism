import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, confusion_matrix
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

def calculate_percentage(df, condition=None, columns=None):
    """
    Calculate the percentage of rows that meet a specific condition or have missing values in specified columns.
    
    Parameters:
    - df: pandas DataFrame
    - condition: A boolean mask for selecting rows based on a condition. For missing values, set to None.
    - columns: List of columns to check for missing values. For conditional percentage, set to None.
    
    Returns:
    - A dictionary with percentages.
    """
    if condition is not None:
        # Calculate the number of rows meeting the condition
        count = df[condition].shape[0]
    elif columns is not None:
        # Calculate the total number of missing values in specified columns
        count = df[columns].isna().sum().sum()
    else:
        return 
    
    # Calculate the total number of rows in the DataFrame
    total_count = df.shape[0]
    
    # Calculate the percentage
    percentage = (count / total_count) * 100
    
    if condition is not None:
        return f'Percentage of rows meeting the condition: {percentage:.2f}%'
    else:
        return {col: round((df[col].isna().sum() / total_count) * 100, 1) for col in columns}


def get_education_level_and_clean_name(row):
    """
    Extracts education level from the 'customer_name' field and cleans the name by removing education abbreviations.

    Args:
        row (pandas.Series): A row from a pandas DataFrame.

    Returns:
        tuple: A tuple containing the education level (as an integer) and the cleaned name (as a string).
    """
    name = row['customer_name']
    # Use case-insensitive searching for broader matching
    if 'Bsc.'.lower() in name.lower():
        education_level = 1
        cleaned_name = name.replace('Bsc.', '').replace('bsc.', '').strip()
    elif 'Msc.'.lower() in name.lower():
        education_level = 2
        cleaned_name = name.replace('Msc.', '').replace('msc.', '').strip()
    elif 'Phd.'.lower() in name.lower():
        education_level = 3
        cleaned_name = name.replace('Phd.', '').replace('phd.', '').strip()
    else:
        education_level = 0
        cleaned_name = name.strip()

    return education_level, cleaned_name

def identify_float_columns_for_conversion(df):
    """
    Identifies float columns in a DataFrame that can potentially be converted to integers
    without losing information. Prints out recommendations for each float column.
    
    Parameters:
    - df: pandas DataFrame to analyze.
    
    """
    # Identify float columns
    float_columns = df.select_dtypes(include=['float']).columns

    # Initialize a list to hold columns that can be converted
    columns_convertible_to_int = []

    # Check for fractional part
    for column in float_columns:
        if not (df[column] % 1).any():  # If the modulo of 1 (fractional part) is 0 for all rows
            columns_convertible_to_int.append(column)
        else:
            None
    # Optionally, return the list of columns that can be converted
    return columns_convertible_to_int


def integer_convert(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Converts multiple columns of a pandas DataFrame to integer data type.

    Parameters:
    - df (pd.DataFrame): A pandas DataFrame.
    - cols (list[str]): A list of strings representing the names of the columns to convert.

    Returns:
    - df (pd.DataFrame): A pandas DataFrame with the columns converted to integer data type.

    """
    #Convert the columns into int64 data type.
    df[cols] = df[cols].astype('int64')
    return df






                                                           
                                                            ## VISUALIZATIONS ##

def customized_histograms(df: pd.DataFrame, cols: list[str]) -> None:
    """
    Plots histograms for specified columns, dropping missing values.

    Parameters:
    - df (pd.DataFrame): The pandas DataFrame with the data to plot.
    - cols (list[str]): A list of strings representing the names of the columns to plot.
    Returns:
    - None
    """
    sns.set_style(style='white')

    # Create subplots
    num_plots = len(cols)
    num_cols = 2
    num_rows = (num_plots + num_cols - 1) // num_cols
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(18, 5*num_rows))
    axs = axs.flatten()

    # Iterate over columns
    for i, col in enumerate(cols):
        # Plot histogram
        ax = axs[i]
        if col == 'customer_age':
            sns.histplot(data=df.dropna(subset=[col]), x=col, ax=ax, bins=10, color='darkseagreen', linewidth=0.5, edgecolor=".2")
            ax.set_title(f'Histogram of {col}', fontsize=14)
        else:
            sns.histplot(data=df.dropna(subset=[col]), x=col, ax=ax, color='darkseagreen', linewidth=0.5, edgecolor=".2")
            ax.set_title(f'Histogram of {col}', fontsize=14)

        # Set axis labels
        ax.set_xlabel(col, fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)

    # Hide empty subplots
    for i in range(num_plots, len(axs)):
        axs[i].set_visible(False)

    # Adjust spacing between subplots
    fig.tight_layout()
    plt.show()


    
    
def customized_bar_charts(data: pd.DataFrame, variables: list[str]) -> None:
    """
    Create customized bar charts.

    Parameters:
    - data (pd.DataFrame): The pandas DataFrame containing the data.
    - variables (list[str]): A list of strings representing the names of the variables to plot.

    Returns:
    - None
    """
    sns.set_style('white')
    
    # Create subplots
    num_plots = len(variables)
    num_cols = 2
    num_rows = (num_plots + num_cols - 1) // num_cols
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(16, 6*num_rows))
    axs = axs.flatten()
    
    # Iterate over columns and plot for each
    for i, var in enumerate(variables):
        ax = axs[i]
        if var == 'customer_tenure':
            unique_years = sorted(data[var].unique())
            data[var].value_counts().loc[unique_years].plot(kind='bar', ax=ax, color='darkseagreen', edgecolor=".2")
            ax.set_xticks(range(len(unique_years)))  # Set ticks at each year
            ax.set_xticklabels(unique_years, fontsize=10, rotation=45)
            # Format tick labels as years
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: '{:.0f}'.format(unique_years[int(x)])))  
            
        else:
            sns.countplot(x=var, data=data, ax=ax, color='darkseagreen', linewidth=0.5, edgecolor=".2")
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=12, rotation=45, ha='right')

        ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
        ax.set_xlabel('')
        ax.set_title(f'{var}')

    # Hide empty subplots
    for i in range(num_plots, len(axs)):
        axs[i].set_visible(False)

    fig.tight_layout()
    plt.show()
 
    
    
def customized_pie_charts(data: pd.DataFrame, variables: list[str]) -> None:
    """
    Create customized pie charts.

    Parameters:
    - data (pd.DataFrame): The pandas DataFrame containing the data.
    - variables (list[str]): A list of strings representing the names of the variables to plot.

    Returns:
    - None
    """
    
    # Create subplots
    num_plots = len(variables)
    num_cols = 2
    num_rows = (num_plots + num_cols - 1) // num_cols
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(14, 5*num_rows))

    axs = axs.flatten()
    
    # Iterate over columns and plot for each
    for i, var in enumerate(variables):
        ax = axs[i]
        counts = data[var].value_counts()
        sizes = counts.values
        labels = counts.index
        wedges, _, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['darkseagreen', 'lightpink'])
        ax.set_title(var, fontsize=14)
        ax.axis('equal')
        ax.legend(loc='upper right', fontsize=10)
        for text in autotexts:
            text.set_fontsize(12)  # Increase font size of percentage values

    for i in range(num_plots, len(axs)):
        axs[i].set_visible(False)

    fig.tight_layout()
    plt.show()
    
     

def filled_line_plot(data: pd.DataFrame, variable: str) -> None:
    """
    Create a filled line plot for a given variable in a DataFrame.

    Parameters:
    - data (pd.DataFrame): The pandas DataFrame containing the data.
    - variable (str): The name of the variable to plot.

    Returns:
    - None
    """

    counts = data[variable].value_counts().sort_index()

    plt.figure(figsize=(10, 6))
    plt.plot(counts.index, counts.values, color='darkseagreen')
    plt.fill_between(counts.index, counts.values, color='darkseagreen')

    plt.xlabel(variable)
    plt.ylabel('Count')
    plt.title(f'Distribution of {variable} Counts')
    plt.xticks(counts.index)
    plt.grid(False)

    plt.show()

   
    
    
def plot_lifetime_spending(data: pd.DataFrame):
    """
    Plot the total lifetime spending by category for columns starting with 'lifetime', excluding 'lifetime_total_distinct_products'.

    Parameters:
    - data (pd.DataFrame): The pandas DataFrame containing the data.

    Returns:
    - None
    """
    # Filter the columns starting with 'lifetime' but exclude 'lifetime_total_distinct_products'
    lifetime_columns = [col for col in data.columns if col.startswith('lifetime') and col != 'lifetime_total_distinct_products']

    # Calculate the total spending for each lifetime spending category
    spending_totals = data[lifetime_columns].sum()

    colors = ['lightblue', 'lightpink', 'lightsalmon', 'orchid', 'purple', 'crimson','brown', 'darkseagreen', 'olive', 'gray']
    spending_totals_sorted = spending_totals.sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    spending_totals_sorted.plot(kind='bar', color=colors)
    plt.xlabel('Lifetime Spending Categories')
    plt.ylabel('Total Spending')
    plt.title('Total Lifetime Spending by Category')
    plt.xticks(rotation=45, ha='right')

    # Set y-axis tick label format to regular numeric format - instead of having le8
    plt.ticklabel_format(style='plain', axis='y')

    plt.tight_layout()
    plt.grid(False)
    plt.show()
    
    
    
def customized_scatter(ax, data, variable1, variable2, color='darkseagreen'):
    """
    Plot a scatter plot between two variables in the given data.

    Args:
        ax (matplotlib.axes.Axes): The axis object of the plot.
        data (pandas.DataFrame): The input data containing the variables.
        variable1 (str): First variable.
        variable2 (str): Second variable.
        color (str, optional): The color of the scatter plot. Defaults to 'darkseagreen'.

    Returns:
        None
    """
    ax.scatter(data[variable1], data[variable2], color=color)  
    
    ax.set_xlabel(variable1)
    
    ax.set_ylabel(variable2)
    
    ax.set_title(f'{variable1} vs {variable2}')

    

def plot_correlation_matrix(data, method):
    """
    Plot a correlation matrix heatmap based on the given data.

    Args:
        data (pandas.DataFrame): The input data for calculating correlations.
        method (str): The correlation method to use (e.g., 'pearson', 'kendall', 'spearman').

    Returns:
        None
    """
    # Filter out columns starting with 'percentage'
    cols_to_include = [col for col in data.columns if not col.startswith('percentage')]
    data_filtered = data[cols_to_include]

    # Select only numeric columns for correlation calculation
    numeric_data = data_filtered.select_dtypes(include=[np.number])

    # Calculate the correlation matrix using the specified method
    corr = numeric_data.corr(method=method)

    # Create a mask to hide the upper triangle of the matrix
    mask = np.tri(*corr.shape, k=0, dtype=bool)
    
    # Set the upper triangle values to NaN
    corr.where(mask, np.NaN, inplace=True)

    # Adjust the width and height of the heatmap as desired
    plt.figure(figsize=(30, 15))

    # Create a custom color map centered on 'darkseagreen'
    cmap = LinearSegmentedColormap.from_list(
        "custom_seagreen", 
        ["white", "darkseagreen", "darkgreen"], 
        N=256
    )

    # Plot the correlation matrix heatmap
    sns.heatmap(corr,
                xticklabels=corr.columns,
                yticklabels=corr.columns,
                annot=True,
                vmin=-1, vmax=1,
                cmap=cmap)

    plt.show()


def high_correlations(data, threshold):
    """
    Calculate correlations between all pairs of numeric variables in the data
    (ignoring columns that start with 'percentage')
    and return correlations above the positive threshold or below the negative threshold,
    sorted in ascending order.
    
    Parameters:
    - data: pandas DataFrame containing the variables
    - threshold: minimum absolute correlation value to consider as strong correlation
    
    Returns:
    - DataFrame containing pairs of variables with correlations
      above the threshold or below its negative, and their corresponding correlation values,
      sorted in ascending order
    """
    numeric_data = data.select_dtypes(include=['number'])
    columns_to_ignore = [col for col in numeric_data.columns if col.lower().startswith('percentage')]
    numeric_data = numeric_data.drop(columns=columns_to_ignore)
    correlations = numeric_data.corr()
    high_correlations = []
    for i, col1 in enumerate(correlations.columns):
        for j, col2 in enumerate(correlations.columns):
            if i < j:  # to avoid duplicate pairs and correlations of variables with themselves
                correlation = correlations.iloc[i, j]
                if correlation >= threshold or correlation <= -threshold:
                    high_correlations.append([col1, col2, correlation])
    high_corr_df = pd.DataFrame(high_correlations, columns=['Variable 1', 'Variable 2', 'Correlation'])
    high_corr_df.sort_values(by='Correlation', inplace=True)
    return high_corr_df   



def plot_k_distance(data, k):
    neigh = NearestNeighbors(n_neighbors=k)
    nbrs = neigh.fit(data)
    distances, indices = nbrs.kneighbors(data)
    
    # Sort the distances to the k-th nearest neighbor
    distances = np.sort(distances[:, k-1], axis=0)
    
    plt.figure(figsize=(10, 6))
    plt.plot(distances,color='darkseagreen')
    plt.xlabel('Points sorted by distance')
    plt.ylabel(f'{k}-th Nearest Neighbor Distance')
    plt.title(f'K-distance Graph for k = {k}')
    plt.show()

def detect_outliers_dbscan(data, eps, min_samples):
    """
    Detect outliers in the data using DBSCAN.

    Parameters:
    data (pd.DataFrame): The dataset for clustering.
    eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    min_samples (int): The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.

    Returns:
    pd.DataFrame: The original DataFrame with an additional column 'outlier' indicating outliers.
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(data)
    
    # Add the cluster labels to the dataframe
    data['cluster'] = dbscan.labels_
    
    # Points labeled as -1 are considered outliers
    data['outlier'] = data['cluster'] == -1
    
    return data

def visualize_outliers_vs_normal(data, outliers, feature1, feature2):
    """
    Visualize the normal data and outliers for two chosen features.

    Parameters:
    data (pd.DataFrame): The full dataset containing both normal data and outliers.
    outliers (pd.DataFrame): The dataset containing only the outliers.
    feature1 (str): The name of the first feature for the x-axis.
    feature2 (str): The name of the second feature for the y-axis.
    """
    plt.figure(figsize=(10, 7))
    plt.scatter(data[feature1], data[feature2], c='darkseagreen', label='Normal Data')
    plt.scatter(outliers[feature1], outliers[feature2], c='lightpink', label='Sampled Outliers')
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.title('Sampled Outliers vs Normal Data')
    plt.legend()
    plt.show()
