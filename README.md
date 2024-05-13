Create a K-means clustering algorithm to group customers of a retail store based on their purchase history.

# Mall Customer Segmentation Analysis
This repository contains Python code for analyzing and visualizing customer segmentation using K-Means clustering on Mall Customers dataset. The code performs the following steps:

1. **Importing and Reading Data**: The Mall Customers dataset (`Mall_Customers.csv`) is imported and read into a pandas DataFrame.

2. **Data Exploration**: The code visualizes the distribution of numerical features (Age, Annual Income, and Spending Score) using seaborn's `displot()` function.

3. **Optimal KMeans Clustering**: The optimal number of clusters for K-Means clustering is determined using the elbow method and silhouette score. The code plots elbow graphs showing inertia depending on the number of clusters and silhouette scores depending on the number of clusters.

4. **K-Means Clustering on Multiple Features**: K-Means clustering is performed on more than two features from the dataset, and elbow graphs are plotted to find the optimal number of clusters.

5. **K-Means Visualization**: The code visualizes the clustering results by plotting scatter plots of pairs of features with different combinations.

## Requirements

- Python 3
- pandas
- matplotlib
- seaborn
- scikit-learn

Install the required libraries using pip:

```
pip install pandas matplotlib seaborn scikit-learn
```

## Usage

1. Clone the repository to your local machine:

```
git clone https://github.com/your_username/mall-customer-segmentation.git
```

2. Navigate to the repository directory:

```
cd mall-customer-segmentation
```

3. Download the dataset (`Mall_Customers.csv`) and place it in the repository directory.

4. Run the Python script:

```
python mall_customer_segmentation.py
```

The script will execute the customer segmentation analysis and visualization, displaying various plots and analysis results.
