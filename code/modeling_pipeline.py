import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
import plotly.express as px
import seaborn as sns
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder



def compare_cluster_solutions(data, cluster1_col, cluster2_col):
    """
    Compare the clusters of two solutions using a confusion matrix.

    Args:
        data (pandas.DataFrame): DataFrame containing cluster assignments for both solutions.
        cluster1_col (str): Column name for the first solution's cluster assignments.
        cluster2_col (str): Column name for the second solution's cluster assignments.

    Returns:
        pandas.DataFrame: Confusion matrix comparing the clusters of the two solutions.
    """
    # Get unique cluster counts
    num_clusters1 = data[cluster1_col].nunique()
    num_clusters2 = data[cluster2_col].nunique()

    # Maximum number of clusters to account for
    max_clusters = max(num_clusters1, num_clusters2)

    # Generate confusion matrix
    conf_matrix = confusion_matrix(data[cluster1_col], data[cluster2_col])

    # Create DataFrame with appropriate labels
    conf_matrix_df = pd.DataFrame(
        conf_matrix,
        index=[f'{cluster1_col} Cluster {i}' for i in range(max_clusters)],
        columns=[f'{cluster2_col} Cluster {i}' for i in range(max_clusters)]
    )

    # Return the relevant subset of the confusion matrix
    return conf_matrix_df.iloc[:num_clusters1, :num_clusters2]


def calculate_group_means(data, group_var, num_features=None):
    """
    Group the data by a specified variable and calculate the mean for each group.

    Args:
        data (pandas.DataFrame): The dataset.
        group_var (str): The variable to group by.
        num_features (int, optional): Number of features to include in the output. If None, include all features.

    Returns:
        pandas.DataFrame: Transposed DataFrame with mean values for each group.
    """
    # Group by the specified variable and compute mean for each group
    grouped_means = data.groupby(group_var).mean()

    # Select specified number of features and transpose the DataFrame
    if num_features is not None:
        grouped_means = grouped_means.iloc[:, :num_features]
    
    transposed_means = grouped_means.T

    # Return the transposed DataFrame
    return transposed_means




def create_dendrogram(data, method, threshold=None):
    """
    Generate a dendrogram for hierarchical clustering.

    Args:
        data (numpy.ndarray or pandas.DataFrame): Dataset for clustering.
        method (str): Linkage method for clustering.
        threshold (float, optional): Threshold value for cutting the dendrogram. Defaults to None.

    Returns:
        None
    """
    # Initialize the AgglomerativeClustering model
    clustering_model = AgglomerativeClustering(linkage=method, distance_threshold=0, n_clusters=None).fit(data)

    # Initialize plot
    fig, ax = plt.subplots()
    plt.title('Hierarchical Clustering Dendrogram')

    # Calculate the number of samples under each node
    sample_counts = np.zeros(clustering_model.children_.shape[0])
    total_samples = len(clustering_model.labels_)

    for idx, nodes in enumerate(clustering_model.children_):
        count = 0
        for node in nodes:
            if node < total_samples:
                count += 1 
            else:
                count += sample_counts[node - total_samples]
        sample_counts[idx] = count

    # Construct the linkage matrix
    linkage_matrix = np.column_stack([clustering_model.children_, clustering_model.distances_, sample_counts]).astype(float)

    # Plot the dendrogram
    dendrogram(linkage_matrix, truncate_mode='level', p=50)

    # Plot a horizontal cut line if a threshold is specified
    if threshold is not None:
        plt.axhline(y=threshold, color='black', linestyle='--')

    # Show the plot
    plt.show()

def visualize_inertia_silhouette(data, min_clusters=2, max_clusters=15):
    """
    Visualize the inertia and silhouette score for various cluster counts.

    Args:
        data (numpy.ndarray or pandas.DataFrame): Dataset for clustering.
        min_clusters (int, optional): Minimum number of clusters to evaluate. Defaults to 2.
        max_clusters (int, optional): Maximum number of clusters to evaluate. Defaults to 15.

    Returns:
        None
    """
    inertia_values = []
    silhouette_scores = []

    cluster_range = range(min_clusters, max_clusters + 1)

    for n_clusters in cluster_range:
        kmeans_model = KMeans(n_clusters=n_clusters, random_state=42).fit(data)
        inertia_values.append(kmeans_model.inertia_)
        labels = kmeans_model.predict(data)
        silhouette_scores.append(silhouette_score(data, labels)) 

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Plot inertia
    axes[0].plot(cluster_range, inertia_values, marker='o')
    axes[0].set_title('Inertia vs. Number of Clusters')
    axes[0].set_xlabel('Number of Clusters')
    axes[0].set_ylabel('Inertia')

    # Plot silhouette score
    axes[1].plot(cluster_range, silhouette_scores, marker='o')
    axes[1].set_title('Silhouette Score vs. Number of Clusters')
    axes[1].set_xlabel('Number of Clusters')
    axes[1].set_ylabel('Silhouette Score')

    plt.tight_layout()
    plt.show()

def plot_dimensionality_reduction_results(transformation, targets):
    """
    Visualize the dimensionality reduction results using a scatter plot.

    Args:
        transformation (numpy.ndarray): The transformed data points after dimensionality reduction.
        targets (numpy.ndarray or list): The target labels or cluster assignments.

    Returns:
        None
    """
    # Convert object labels to categorical variables
    labels, targets_categorical = np.unique(targets, return_inverse=True)

    # Create a scatter plot of the t-SNE output
    cmap = plt.cm.tab10
    norm = plt.Normalize(vmin=0, vmax=len(labels) - 1)
    plt.scatter(transformation[:, 0], transformation[:, 1], c=targets_categorical, cmap=cmap, norm=norm)

    # Create a legend with the class labels and corresponding colors
    handles = [plt.scatter([], [], c=cmap(norm(i)), label=label) for i, label in enumerate(labels)]
    plt.legend(handles=handles, title='Clusters')

    # Display the plot
    plt.show()

def plot_cluster_by_coordinates(df,cm,umap,cluster_label, x_min_gb, x_max_gb, y_min_gb, y_max_gb):
     # Map the cluster label to its corresponding name
    cluster_name = cm.get(cluster_label, f'Cluster {cluster_label}')
    
    # Filter the embedding and labels to the region of interest
    region_mask_gb = (umap[:, 0] >= x_min_gb) & (umap[:, 0] <= x_max_gb) & (umap[:, 1] >= y_min_gb) & (umap[:, 1] <= y_max_gb)
    region_embedding_gb = umap[region_mask_gb]
    region_labels_gb = df['mm_ward9'][region_mask_gb]
    region_data_gb = df[region_mask_gb]  # Extract corresponding rows from the original data

    # Filter the region embedding to only include the specific cluster
    region_mask = (region_labels_gb == cluster_label)
    region_data = region_data_gb[region_mask]
    print(f"Number of points in {cluster_name} in the region:", len(region_data))
    
    # Set sample size
    sample_size = min(100, len(region_data))  # Adjust sample size if the cluster is smaller than 100
    
    # Take a random sample from each cluster
    sample = region_data.sample(n=sample_size, random_state=42)
    
    # Reset index to make 'customer_id' a column again
    sample = sample.reset_index()
    cluster_location = sample.loc[:, ['customer_id', 'latitude', 'longitude']].copy()
    
    # Plotly scatter plot on a map
    fig = px.scatter_mapbox(cluster_location, lat="latitude", lon="longitude", hover_name="customer_id",
                            zoom=6, height=400, title=f"{cluster_name.capitalize()} Geographic Distribution")
    fig.update_layout(mapbox_style="open-street-map", margin=dict(l=0, r=0, t=30, b=0))
    fig.show()


def check_for_karens(data, method_column, cluster_number):
    """
    Check if there are any rows in the data where the number of complaints is greater than 4
    and the specified cluster column equals the given cluster number.

    Parameters:
    - data: pd.DataFrame, the dataset to check.
    - method_column: str, the name of the column used for clustering.
    - cluster_number: int, the cluster number to check.

    Returns:
    - A DataFrame containing the rows that match the criteria.
    """
    # Filter rows where 'number_complaints' > 4
    high_complaints = data[data['number_complaints'] > 4]

    # Check if these rows are in the specified cluster
    high_complaints_in_cluster = high_complaints[high_complaints[method_column] == cluster_number]

    return high_complaints_in_cluster



def get_r2_hc(df, link_method, max_nclus=6, min_nclus=1, dist="euclidean"):
    """This function computes the R2 for a set of cluster solutions given by the application of a hierarchical method.
    The R2 is a measure of the homogenity of a cluster solution. It is based on SSt = SSw + SSb and R2 = SSb/SSt.

    Parameters:
    df (DataFrame): Dataset to apply clustering
    link_method (str): either "ward", "complete", "average", "single"
    max_nclus (int): maximum number of clusters to compare the methods
    min_nclus (int): minimum number of clusters to compare the methods. Defaults to 1.
    dist (str): distance to use to compute the clustering solution. Must be a valid distance. Defaults to "euclidean".

    Returns:
    ndarray: R2 values for the range of cluster solutions
    """
    def get_ss(df):
        ss = np.sum(df.var() * (df.count() - 1))
        return ss  # return sum of sum of squares of each df variable

    sst = get_ss(df)  # get total sum of squares

    r2 = []  # where we will store the R2 metrics for each cluster solution

    for i in range(min_nclus, max_nclus+1):  # iterate over desired ncluster range
        cluster = AgglomerativeClustering(n_clusters=i, metric=dist, linkage=link_method)

        # get cluster labels
        hclabels = cluster.fit_predict(df)

        # concat df with labels
        df_concat = pd.concat((df, pd.Series(hclabels, name='labels', index=df.index)), axis=1)

        # compute ssw for each cluster labels
        ssw_labels = df_concat.groupby(by='labels').apply(get_ss)

        # remember: SST = SSW + SSB
        ssb = sst - np.sum(ssw_labels)

        r2.append(ssb / sst)  # save the R2 of the given cluster solution

    return np.array(r2)


def plot_r2_hc(data, max_nclus=6, min_nclus=1, dist="euclidean"):
    # Prepare input
    hc_methods = ["ward", "complete", "average", "single"]

    # Call function defined above to obtain the R2 statistic for each hc_method
    r2_hc_methods = np.vstack(
        [
            get_r2_hc(df=data, link_method=link, max_nclus=max_nclus, min_nclus=min_nclus, dist=dist)
            for link in hc_methods
        ]
    ).T
    r2_hc_methods = pd.DataFrame(r2_hc_methods, index=range(1, max_nclus + 1), columns=hc_methods)

    # Plot data
    fig = plt.figure()
    sns.lineplot(data=r2_hc_methods, linewidth=2.5, markers=["o"]*4)

    # Finalize the plot
    fig.suptitle("R2 plot for various hierarchical methods", fontsize=21)
    plt.gca().invert_xaxis()  # invert x axis
    plt.legend(title="HC methods", title_fontsize=11)
    plt.xticks(range(1, max_nclus + 1))
    plt.xlabel("Number of clusters", fontsize=13)
    plt.ylabel("R2 metric", fontsize=13)

    plt.show()



#Association Rules

def generate_association_rules(customers, basket, join_column='customer_id', list_column='list_of_goods',
                              min_support=0.2, metric='lift', min_threshold=1):
    """
    Perform the association rules on customer-basket data.

    Args:
        customers (pandas.DataFrame): Customer data.
        basket (pandas.DataFrame): Basket data.
        join_column (str): Column used for joining the customer and basket data.
        list_column (str): Column that contains the list of goods in the basket.
        min_support (float): The minimum support threshold for generating frequent itemsets.
        metric (str): Metric used for evaluating association rules.
        min_threshold (float): Minimum threshold for the metric to consider a rule.

    Returns:
        pandas.DataFrame: Generated association rules.
    """
    # Join basket and customer data on the specified column
    data = pd.merge(basket, customers, on=join_column, how='inner')

    # Extract the transactions from the joined data
    transactions = data[list_column].apply(lambda x: [item.strip() for item in x[1:-1].split(',')])

    # Convert transactions to a transaction matrix
    transaction_encoder = TransactionEncoder()
    te_fit = transaction_encoder.fit(transactions).transform(transactions)
    transaction_items = pd.DataFrame(te_fit, columns=transaction_encoder.columns_)

    # Generate frequent itemsets using Apriori algorithm
    frequent_itemsets = apriori(transaction_items, min_support=min_support, use_colnames=True)

    # Generate association rules from frequent itemsets
    rules = association_rules(frequent_itemsets, metric=metric, min_threshold=min_threshold)

    return rules

































