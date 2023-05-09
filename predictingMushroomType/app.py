from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline
import pandas as pd
from pyspark.ml.stat import ChiSquareTest
from pyspark.sql.functions import when, col
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, GBTClassifier, NaiveBayes
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import numpy as np
import json
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

# How to start spark standalone cluster and worker node
# 1) Go to /Applications/spark-3.4.0-bin-hadoop3
# 2) run ->  ./sbin/start-master.sh  
# 3) access web ui -> http://localhost:8080/
# 4) Start worker node -> ./sbin/start-worker.sh spark://Justins-MacBook-Pro-7.local:7077
# 5) run -> ./bin/spark-submit \
#           --master spark://Justins-MacBook-Pro-7.local:7077 \
#           /Users/justinkim/Documents/CSUF/Spring2023/CPSC531/finalProjectMongodb/predictingMushroomType/app.py \
#           1000

# Create new SparkSession with app name ml-mushroomsType
spark = SparkSession.builder.appName('ml-mushroomsType').getOrCreate()

# URI for mongodb database connection
uri = "mongodb+srv://mushroomAdmin:DrLUCn6a96FJPk0T@mushroomdata.6jqgppn.mongodb.net/?retryWrites=true&w=majority"

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

# Access mushroom collection in cpsc531 database
db = client['cpsc531_FinalProject']
mushroomCollection = db['mushrooms']
cursor = mushroomCollection.find({}, {'_id': 0})
list_cur = list(cursor)

# Create spark dataframe from collection
df = spark.createDataFrame(list_cur)

# See the datatype of each column
df.printSchema()

# See the first 20 rows
df.show(20)

# veil type has only 1 unique value so drop as it is not informative
df = df.drop('veil-type')

# get the list of columns that are categorical
categorical_cols = df.columns

# remove 'class' column since that is our dependent column
categorical_cols.remove('class')

# Create a list of 'StringIndexer' objects which turns the categorical columns into numerical indices
string_indexers = [StringIndexer(inputCol=col, outputCol=f"{col}Index", handleInvalid='skip') for col in
                   categorical_cols]

# Convert the StringIndexer indices into one-hot encoded vectors to prevent numerical values from being interpreted as
# having an order or rank
encoder = OneHotEncoder(inputCols=[f"{col}Index" for col in categorical_cols],
                        outputCols=[f"{col}Encoded" for col in categorical_cols])

# Assemble the one-hot encoded vectors into a single 'feature' vector
assembler = VectorAssembler(inputCols=encoder.getOutputCols(), outputCol='features')

# Create a Pipeline object which will apply the sequence of transformations to the dataframe
pipeline = Pipeline(stages=[*string_indexers, encoder, assembler])

# Apply the sequence of transformations to df
model = pipeline.fit(df)
df.write.option("header",True).option("delimiter",",").csv("mushrooms_df", mode='overwrite')
model.write().overwrite().save("pModel")
df_encoded = model.transform(df)

# Encode the 'class' attribute so that it is numerical: 1 means it is poisonous and 0 means it is not
df_encoded = df_encoded.withColumn('label', when(col('class') == 'p', 1).otherwise(0))

# Show transformed dataframe
df_encoded.show(vertical=True)
pandas_df = df_encoded.toPandas()
print(pandas_df)

# Show the new dataframe
df_encoded_final = df_encoded.select('cap-shapeEncoded', 'cap-surfaceEncoded', 'cap-colorEncoded',
                                     'bruisesEncoded', 'odorEncoded', 'gill-attachmentEncoded', 'gill-spacingEncoded',
                                     'gill-sizeEncoded', 'gill-colorEncoded', 'stalk-shapeEncoded', 'stalk-rootEncoded',
                                     'stalk-surface-above-ringEncoded', 'stalk-surface-below-ringEncoded',
                                     'stalk-color-above-ringEncoded', 'stalk-color-below-ringEncoded',
                                     'veil-colorEncoded', 'ring-numberEncoded', 'ring-typeEncoded',
                                     'spore-print-colorEncoded', 'populationEncoded', 'habitatEncoded', 'features',
                                     'label')

# Loop through categorical column to calculate the chi-squared test statistics between each categorical attribute
# and the target attribute 'label' which is 'class'
# If the p-value is <= 0.05 there is a significant association between the class variable and the other variable
for col in categorical_cols:
    results = ChiSquareTest.test(df_encoded_final, f"{col}Encoded", 'label').head()
    # get the p-value result
    p_value = results.pValues[0]
    # print out the results
    print(f"Chi-squared test for {col}:")
    print(f"  P-value: {p_value}")

(training, testing) = df_encoded_final.randomSplit([0.7, 0.3])

print("\n")
# Using Logistic Regression for machine learning
lr = LogisticRegression(featuresCol='features', labelCol='label')
lr_model = lr.fit(training)
# Save model to file
lr_model.write().overwrite().save('lr_model')

lr_predictions = lr_model.transform(testing)
lr_predictions.show()
lr_evaluator = BinaryClassificationEvaluator(rawPredictionCol='rawPrediction', labelCol='label')
lr_accuracy = lr_evaluator.evaluate(lr_predictions)
print(f"Logistic Regression Accuracy: {lr_accuracy}")

print("\n")
# Decision Tree Classifier
dt = DecisionTreeClassifier(featuresCol='features', labelCol='label')
dt_model = dt.fit(training)
# Save model to file
dt_model.write().overwrite().save('dt_model')

dt_predictions = dt_model.transform(testing)
dt_evaluator = BinaryClassificationEvaluator(rawPredictionCol='rawPrediction', labelCol='label')
dt_accuracy = dt_evaluator.evaluate(dt_predictions)
print(f"Decision Tree Classification Accuracy: {dt_accuracy}")
print(dt_model.toDebugString)

print("\n")
# Random Forest Classifier
rf = RandomForestClassifier(featuresCol='features', labelCol='label')
rf_model = rf.fit(training)
# Save model to file
rf_model.write().overwrite().save('rf_model')

rf_predictions = rf_model.transform(testing)
rf_evaluator = BinaryClassificationEvaluator(rawPredictionCol='rawPrediction', labelCol='label')
rf_accuracy = rf_evaluator.evaluate(rf_predictions)
print(f"Random Forest Classification Accuracy: {rf_accuracy}")
importantAttr = dict(zip(training.columns, rf_model.featureImportances))
importantAttr = sorted(importantAttr.items(), key=lambda x: x[1], reverse=True)
importantAttr = json.dumps(importantAttr, indent=3)
print(importantAttr)

print("\n")
# Gradient-Boosted Tree Classifier
gbt = GBTClassifier(featuresCol='features', labelCol='label')
gbt_model = gbt.fit(training)
# Save model to file
gbt_model.write().overwrite().save('gbt_model')

gbt_predictions = gbt_model.transform(testing)
gbt_evaluator = BinaryClassificationEvaluator(rawPredictionCol='rawPrediction', labelCol='label')
gbt_accuracy = gbt_evaluator.evaluate(gbt_predictions)
print(f"Gradient-Boosted Tree Classification Accuracy: {gbt_accuracy}")
importantAttr = dict(zip(training.columns, gbt_model.featureImportances))
importantAttr = sorted(importantAttr.items(), key=lambda x: x[1], reverse=True)
importantAttr = json.dumps(importantAttr, indent=3)
print(importantAttr)


print("\n")
# Naive Bayes Classifier
nb = NaiveBayes(featuresCol='features', labelCol='label')
nb_model = nb.fit(training)
# Save model to file
nb_model.write().overwrite().save('nb_model')

nb_predictions = nb_model.transform(testing)
nb_evaluator = BinaryClassificationEvaluator(rawPredictionCol='rawPrediction', labelCol='label')
nb_accuracy = nb_evaluator.evaluate(nb_predictions)
print(f"Naive Bayes Classification Accuracy: {nb_accuracy}")
print(nb_model.theta.toArray())
print("\n")
print("\n")

spark.stop()
