from flask import Flask, request, jsonify, render_template
from pyspark.ml import PipelineModel, Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.sql.functions import lit, col
from pyspark.sql import SparkSession, Row
from pyspark.ml.classification import LogisticRegressionModel, DecisionTreeClassificationModel, \
    RandomForestClassificationModel, GBTClassificationModel, NaiveBayesModel
from pyspark.sql.types import StructType, StructField, StringType
import json
import pandas as pd
import pickle
from pyspark.sql.functions import udf

spark = SparkSession.builder.appName('myApp').getOrCreate()
model = PipelineModel.load("./pModel")

lr_model = LogisticRegressionModel.load('./lr_model')
dt_model = DecisionTreeClassificationModel.load('./dt_model')
gbt_model = GBTClassificationModel.load('./gbt_model')
nb_model = NaiveBayesModel.load('./nb_model')
rf_model = RandomForestClassificationModel.load('./rf_model')

df_original = spark.read.csv('./mushrooms_df', header=True, inferSchema=True)

app = Flask(__name__)


@app.route("/")
def main_page():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    predict_data = request.form.to_dict()
    predict_data['class'] = str('e')

    # Create a new Row object with the data
    new_row = Row(**predict_data)
    new_row= Row(*(new_row[col_name] for col_name in df_original.columns))

    # Create a new DataFrame with the new row
    new_row_df = spark.createDataFrame([new_row], schema=df_original.schema)
    new_row_df.coalesce(1).write.csv('newMushroomRow', header=True, mode='overwrite')

    # Transform data
    df_encoded = model.transform(new_row_df)

    # make prediction
    lr_prediction = lr_model.transform(df_encoded)
    dt_prediction = dt_model.transform(df_encoded)
    gbt_prediction = gbt_model.transform(df_encoded)
    nb_prediction = nb_model.transform(df_encoded)
    rf_prediction = rf_model.transform(df_encoded)

    # combine predictions into a single DataFrame
    all_predictions = lr_prediction.select("prediction").union(dt_prediction.select("prediction")).union(gbt_prediction.select("prediction")).union(nb_prediction.select("prediction")).union(rf_prediction.select("prediction"))
    all_predictions.coalesce(1).write.csv('predictionsCombined', mode='overwrite')

    # group by prediction value and count the number of occurrences
    counts = all_predictions.groupBy("prediction").agg({"prediction": "count"}).withColumnRenamed("count(prediction)", "count")
    counts.write.csv('predictionResult', mode='overwrite')
    # get the most common prediction value
    most_common = counts.orderBy(col("count").desc()).first()[0]

    # return the result as a string
    if most_common == 1:
        return render_template('poisonous.html')
    elif most_common is None:
        return render_template('unknown.html')
    else:
        return render_template('edible.html')
    

@app.route("/chart")
def chart_page():
    return render_template('chart.html')