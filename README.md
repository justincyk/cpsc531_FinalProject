# Spring 2023 CPSC 531-03 Final Project Implementaion
### Member: Justin Kim

##### * main branch contains ML Python App that uses local storage Mushroom.csv while mongodbBranch contains ML Python App that uses Mongodb database to access Mushroom collection *

## Project Functionality, Architecture & Design
### 3 Parts:
   - Predicting Mushroom Type ( Poisonous or Edible ) Python Application using mushroom.csv from local storage and PySpark ML models
   - Predicting Mushroom Type ( Poisonous or Edible ) Python Application using Mongodb database collection and Pyspark ML models
   - Mushroom Type Prediction Web Application using Flask, HTML, CSS, and saved PySpark ML models

#### * Both option 1 and 2 are run on Spark Standalone Cluster by submitting app.py as a job *


### 1) Mushroom Prediction Python Application using Mongodb
#### Used: PyMongo MongoClient, PyMongo ServerApi, Pyspark sparkSession, PySpark Dataframe, PySpark Machine learning Models, PySpark features - StringIndexer, OneHotEncoder, Vector Assembler, Pyspark Pipeline, PySpark BinaryClassificationEvaluator, Pandas
   - Connects to Mongodb using Pymongo server and client to access cpsc531_FinalProject database mushroom collection
   - Creates sparkSession called 'ml-mushroomType'
   - Converts Mongodb mushroom collection to Spark Dataframe using sparkSession createDataFrame method
   - Performs data analysis and transformation on dataframe using StringIndexer, OneHotEncoder, and VectorAssembler
   - Splits dataframe into training and testing dataframes
   - Fits and transforms training dataframe on Machine learning models ( Logistic Regression, Decision Tree Classifer, Random Forest Classifier, Gradient Boosting, Naive Bayes )
   - ML Models predict Mushroom type on test dataframe
   - Compares the prediction result with actual Mushroom type values to get prediction accuracy
   - Finds the most important attributes from the machine learning models
   - Saves fitted machine learning models to use for web application


### 2) Mushroom Prediction Python Application using local storage
#### Used: Pyspark sparkSession, PySpark Dataframe, PySpark Machine learning Models, PySpark features - StringIndexer, OneHotEncoder, Vector Assembler, Pyspark Pipeline, PySpark BinaryClassificationEvaluator, Pandas
   - Creates sparkSession called 'ml-mushroomType'
   - Retrieves mushroom.csv file from local storage
   - Converts mushroom.csv file to Spark Dataframe using sparkSession
   - Performs data analysis and transformation on dataframe
   - Splits dataframe into training and testing dataframes
   - Fits and transforms training dataframe on Machine learning models (Logistic Regression, Decision Tree Classifer, Random Forest Classifier, Gradient Boosting, Naive Bayes)
   - ML Models predict Mushroom type on test dataframe
   - Compares the prediction result with actual Mushroom type values to get prediction accuracy
   - Finds the most important attributes from the machine learning models
   - Saves fitted machine learning models to use for web application


### 3) Mushroom Prediction Web Application
#### Used: Flask, HTML, CSS, saved PySpark Machine Learning models, PySpark sparkSession, PySpark Pipeline, Mongodb
   - User fills out a form about the characteristics of a mushroom they want to know about
   - User submits data form once they finished filling out the form
   - Web Application will then refresh and inform user if the mushroom is poisonous or not
   - Website can also display analytical graphs fetched from Mongodb mushroom collection by user going on '/chart' 
   - Website form was created using HTML and CSS
   - Once user submits their data, data will be sent to Flask
   - Flask will read the data, convert data to dictionary, create and start sparkSession, convert dictioanry to a PySpark Dataframe using Mushroom dataframe schema
   - Flask will then transform the dataframe using saved Pyspark Pipeline model
   - Once dataframe is transformed, Flask will run the user's dataframe on the five saved ML models
   - The majority result/prediction from the five ML models will be outputted to the user using HTML and CSS
   - Flask also directs users that go on '/chart' to a html page that renders data graphs that are fetched and continuosly updated from Mongodb database



## Project Deployment Instructions ( On Apple Macbook )
1. Clone GitHub repository to local computer
2. Access cloned GitHub repository on terminal
3. Once in directory of cloned GitHub repository, type and enter `source bin/activate` to start venv environment
4. Once venv environment is started, type and enter `pip3 install -r requirements.txt` to download necessary packages to run Python ML Mushroom Type Prediction Application and Mushroom Type Prediction Web Application



## Steps to Run the Application
1. To run Python ML Mushroom Type Prediction Application without Spark Standalone Cluster using Mushroom.csv local file
   -  Go to `./predictingMushroomType` folder
   -  Type and enter `python3 app.py` to run application
   -  All saved ML models in the current directory will be accessed and rewritten every time application is run
   -  Results from application will be printed to terminal
2. To run Python ML Mushroom Type Prediction Application without Spark Standalone Cluster using Mongodb Database
   -  Change Git branch by typing `git checkout mongodbBranch`
   -  Go to `./predictingMushroomType` folder
   -  Type and enter `python3 app.py` to run application
   -  Mushroom data will be retrieved by accessing Mongodb database through PyMongo
   -  All saved ML models in the current directory will be accessed and rewritten every time application is run
3. To run Python ML Mushroom Type Prediction Application on Spark Standalone Cluster using Mongodb Database
   - Go to current directory Spark ( Ex. `cd /Applications/spark-3.4.0-bin-hadoop3` )
   - Start Master Node ( Ex. `./sbin/start-master.sh` )
   - Start Worker Node and provide address of Master Node ( Ex. `./sbin/start-worker.sh spark://Justins-MacBook-Pro-7.local:7077` )
   - Run Spark Job and provide address of master node and location/file of Python App to run
    ( Ex. `./bin/spark-submit \
           --master spark://Justins-MacBook-Pro-7.local:7077 \
           /Users/justinkim/Documents/CSUF/Spring2023/CPSC531/finalProjectMongodb/predictingMushroomType/app.py \
           1000` )
   - Check status of job by accessing web ui ( Ex. `http://localhost:8080/` )
   - End Worker Node
   - End Master Node
 4. To run Mushroom Type Prediction Web Application
   - Go to directory `./mushroomWebApp`
   - Type and enter `Flask run`
   - Access web application by going on `http://127.0.0.1:5000`
   - Fill out the form according to your mushroom's characteristics
   - Submit form once form is completed
   - Web Application will refresh and inform you if the Mushroom is poisonous or not
   - Go to `http://127.0.0.1:5000/chart` to access data analysis graphs fetched from Mongodb database
   - Press control+c on terminal to end Flask session



## Test Results
   1. ML Logistic Regression Prediction Model Accuracy: 
   2. ML Decision Tree Classifier Prediction Model Accuracy:
   3. ML Random Forest Classifier Prediction Model Accuracy:
   4. ML Gradient Boosting Prediction Model Accuracy:
   5. ML Naive Bayes Prediction Model Accuracy:


