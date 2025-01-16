Yelp-Review-ML-Model

- Description
  - Howdy! This is a personal project using a machine learning logistic regression model to analyze a Yelp data set including customer reviews.
    The model is used to and categorize, count, and visualize the distributions into three sets: positive, neutral, and negative. 
    The purpose of this project is to display conceptual understanding using basic machine learning tool application.

- Installation Instructions
  - Dowload YelpMLProject.py from this repository.
  - Open file in an IDE ex: Spyder, VS Code, etc..
  - Install libraries for pandas matplotlib, seaborn, and scikit-learn
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

1. Sentiment Counts:
 <table>
  <thead>
    <tr>
      <th>Sentiment</th>
      <th>Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Positive</td>
      <td>69,729</td>
    </tr>
    <tr>
      <td>Negative</td>
      <td>18,909</td>
    </tr>
    <tr>
      <td>Neutral</td>
      <td>11,362</td>
    </tr>
  </tbody>
</table>

2. Classification Report:
<table>
  <thead>
    <tr>
      <th>Sentiment</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F1-Score</th>
      <th>Support</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Negative</td>
      <td>0.81</td>
      <td>0.81</td>
      <td>0.81</td>
      <td>3,762</td>
    </tr>
    <tr>
      <td>Neutral</td>
      <td>0.54</td>
      <td>0.30</td>
      <td>0.38</td>
      <td>2,275</td>
    </tr>
    <tr>
      <td>Positive</td>
      <td>0.90</td>
      <td>0.96</td>
      <td>0.93</td>
      <td>13,963</td>
    </tr>
  </tbody>
  <tfoot>
    <tr>
      <td><strong>Accuracy</strong></td>
      <td colspan="4">0.86 (20,000 samples)</td>
    </tr>
    <tr>
      <td><strong>Macro Avg</strong></td>
      <td>0.75</td>
      <td>0.69</td>
      <td>0.71</td>
      <td>20,000</td>
    </tr>
    <tr>
      <td><strong>Weighted Avg</strong></td>
      <td>0.84</td>
      <td>0.86</td>
      <td>0.85</td>
      <td>20,000</td>
    </tr>
  </tfoot>
</table>

    
![SentimentHeatMap](https://github.com/user-attachments/assets/190e6b29-b9f9-40d2-9386-1bcd9e139df3)
![SentimentBarDistrubution](https://github.com/user-attachments/assets/7367a292-03dc-4e14-be9a-623676903c7c)


**Contact Information:**

Colby Swayze  
Email: [colbyswayze@tamu.edu]  
Phone: (949) 424-4694

    
