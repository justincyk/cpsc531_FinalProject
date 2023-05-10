# Spring 2023 CPSC 531-03 Final Project Implementaion
### Member: Justin Kim


## Project Functionality, Architecture & Design
### 3 Parts:
   - Predicting Mushroom Type ( Poisonous or Edible ) Python Application using mushroom.csv from local storage and PySpark ML models
   - Predicting Mushroom Type ( Poisonous or Edible ) Python Application using Mongodb database collection and Pyspark ML models
   - Mushroom Type Prediction Web Application using Flask, HTML, CSS, and saved PySpark ML models

#### * Both option 1 and 2 are run on Spark Standalone Cluster by submitting app.py as a job *

### 1) Mushroom Prediction Python Application using Mongodb (PyMongo MongoClient, PyMongo ServerApi, Pyspark sparkSession, PySpark Dataframe, PySpark Machine learning Models, PySpark features - StringIndexer, OneHotEncoder, Vector Assembler, Pyspark Pipeline, PySpark BinaryClassificationEvaluator, Pandas)
   - Connects to Mongodb using Pymongo server and client to access cpsc531_FinalProject database mushroom collection
   - Create sparkSession called 'ml-mushroomType'
   - Convert Mongodb mushroom collection to Spark Dataframe using sparkSession createDataFrame method
   - Perform data analysis and transformation on dataframe using StringIndexer, OneHotEncoder, and VectorAssembler
   - Split dataframe into training and testing dataframes
   - Fit and transform training dataframe on Machine learning models (Logistic Regression, Decision Tree Classifer, Random Forest Classifier, Gradient Boosting, Naive Bayes)
   - Predict Mushroom type on test dataframe and compare the prediction value with actual value to get prediction accuracy
   - Find the most important attributes from the machine learning models
   - Save fitted machine learning models to use for web application


### 2) Mushroom Prediction Python Application using local storage (Pyspark sparkSession, PySpark Dataframe, PySpark Machine learning Models, PySpark features - StringIndexer, OneHotEncoder, Vector Assembler, Pyspark Pipeline, PySpark BinaryClassificationEvaluator, Pandas)
   - Create sparkSession called 'ml-mushroomType'
   - Retrieve mushroom.csv file from local storage
   - Convert mushroom.csv file to Spark Dataframe using sparkSession
   - - Perform data analysis and transformation on dataframe
   - Split dataframe into training and testing dataframes
   - Fit and transform training dataframe on Machine learning models (Logistic Regression, Decision Tree Classifer, Random Forest Classifier, Gradient Boosting, Naive Bayes)
   - Predict Mushroom type on test dataframe and compare the prediction value with actual value to get prediction accuracy
   - Find the most important attributes from the machine learning models
   - Save fitted machine learning models to use for web application


### 3) Mushroom Prediction Web Application (Flask, HTML, CSS, saved PySpark Machine Learning models, PySpark sparkSession, PySpark Pipeline, Mongodb )
   - User fills out a form about the characteristics of a mushroom they want to know about
   - User submits data form once they finished filling out the form
   - Website will then refresh and inform user if the mushroom is poisonous or not
   - Website can also display analytical graphs fetched from Mongodb mushroom collection by going on '/chart' 
   - Website form was created using HTML and CSS
   - Once user submits their data, data will be sent to Flask
   - Flask will read the data, convert data to dictionary, start sparkSession, convert dictioanry to a PySpark Dataframe using the mushroom dataframe schema
   - Flask will then transform the dataframe using saved Pyspark Pipeline model
   - Once dataframe is transformed, Flask will run the dataframe on the five saved ML models
   - The majority result/prediction from the five ML model will be outputted to the user using HTML and CSS
   - Flask also directs users that go on '/chart' to a html page that renders data graphs that are fetched and continuosly updated from Mongodb database
