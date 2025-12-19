# reasons_for_absenteeism
Using data mining techniques such as clustering to explore reasons for employee absence


# About
Employee absenteeism poses a significant challenge for organizations, as it directly affects productivity, operational efficiency, and workforce planning. This project applies unsupervised learning techniques to analyze absenteeism patterns within a Brazilian courier company experiencing rising absence rates. The analysis is based on a dataset of 800 observations and 21 anonymized variables, covering personal characteristics, work conditions, commuting factors, and health-related indicators.

The project followed a structured data science workflow, including exploratory data analysis, data cleaning, feature engineering, scaling, and clustering. Multiple clustering approaches were tested, including K-Means and hierarchical clustering with different scaling techniques. Model selection was guided by inertia, Silhouette scores, dendrogram analysis, and visual validation using UMAP, applied exclusively for visualization purposes.

The final solution employed hierarchical clustering with Ward’s linkage, resulting in ten distinct employee segments. These clusters revealed clear differences in absenteeism behavior driven by structural factors such as commuting distance, career stage, and physical job demands, as well as behavioral patterns linked to lifestyle and engagement. The results highlight that absenteeism is not a uniform phenomenon but varies substantially across employee profiles.
Overall, the findings demonstrate the value of employee segmentation for absenteeism management and provide actionable insights to support targeted HR interventions, preventive strategies, and workforce planning decisions


# Organization  
  
REASONS_FOR_ABSENTEEISM/  
│  
├── code/
│   ├── data_exploration_and_preprocessing.ipynb   # Exploring, visualizing, and pre-processing the raw data  
│   ├── clustering.ipynb                           # Clustering notebook  
│   ├── association_rules.ipynb                    # After clustering, creating association rules based on findings  
│   ├── utils1.py                                  # Data pre-processing and feature transformation utils  
│   ├── utils2.py                                  # Segmentation utils  
│   └── __pycache__/  
│  
├── data/
│   ├── absenteeism_data.csv                       # Raw data  
│   ├── Processed_Absenteeism_Dataset.csv          # Clean data, ready for clustering  
│   ├── Final_Clustered_Worker_Data.csv            # Clean data, with cluster associated  
│   ├── Grouped Workers Absenteeism Dataset.csv    # Data grouped by worker_id  
│   └── Worker_Association_Rules.csv               # Association rules data mined after clustering  
│  
│  
├── README.md  
└── requirements.txt  

