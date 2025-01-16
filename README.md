Yelp-Review-ML-Model

- Description
  - Howdy! This is a personal project using a machine learning regression model to analyze a Yelp data set including customer reviews.
    The model is used to and categorize, count, and visualize the distributions into three sets: positive, neutral, and negative. 
    The purpose of this project is to display conceptual understanding using basic machine learning tool application.

- Installation Instructions
  - Dowload YelpMLProject.py from this repository.
  - Open file in an IDE ex: Spyder, VS Code, etc..
  - Visit https://www.yelp.com/dataset/download and dowload the .json file.
  
- How to Use
  - Open file in an IDE ex: Spyder, VS Code, etc..
  - ensure the file path to the yelp_dataset/yelp_academic_dataset_review.json can be located by your IDE on line 147.
     dataset_source = "C:/Users/admin/Downloads/yelp_dataset/yelp_academic_dataset_review.json"
  - Confirm the file path in the code matches with your device.
  - Change code to preffered chunk size on line 19.
    def load_data_in_chunks(source, chunksize=100000, max_chunks=1):
  - Run the file.
  - Example graphed visualizatoins of data and output using a chunksize of 100,000 from the file.
    
    Sentiment Counts:
    sentiment
    Positive    69729
    Negative    18909
    Neutral     11362
 
    Evaluation complete. Classification Report:
              precision    recall  f1-score   support

    Negative      0.81      0.81      0.81      3762
    Neutral       0.54      0.30      0.38      2275
    Positive      0.90      0.96      0.93     13963

    accuracy                           0.86     20000
    macro avg      0.75      0.69      0.71     20000
    weighted avg   0.84      0.86      0.85     20000
    
    ![SentimentHeatMap](https://github.com/user-attachments/assets/6019011a-728b-40fb-8c20-85e4b7696026)
    ![SentimentBarDistrubution](https://github.com/user-attachments/assets/574de184-8a9a-4559-8093-1028b7c006fa)

**Contact Information:**

Colby Swayze  
Email: [colbyswayze@tamu.edu]  
Phone: (949) 424-4694

    
