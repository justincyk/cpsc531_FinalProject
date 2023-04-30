from flask import Flask, request, jsonify, render_template
from pyspark.ml import PipelineModel, Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.sql.functions import lit, col
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegressionModel, DecisionTreeClassificationModel, \
    RandomForestClassificationModel, GBTClassificationModel, NaiveBayesModel
from pyspark.sql.types import StructType, StructField, StringType
import json
import pandas as pd
import pickle

spark = SparkSession.builder.appName('myApp').getOrCreate()
model = PipelineModel.load("./pModel")

lr_model = LogisticRegressionModel.load('./lr_model')
df_original = spark.read.csv('./mushrooms_df', header=True, inferSchema=True)

app = Flask(__name__)


@app.route("/")
def hello():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    predict_data = request.form.to_dict()
    # # convert dictionary to JSON string and then to RDD
    # rdd = spark.sparkContext.parallelize([json.dumps(predict_data)])
    # # create DataFrame from RDD
    # df = spark.read.json(rdd)
    # df = df.withColumn('class', lit('e'))
    # df_reordered = df.select(
    # col('class'),
    # col('cap-shape'),
    # col('cap-surface'),
    # col('cap-color'),
    # col('bruises'),
    # col('odor'),
    # col('gill-attachment'),
    # col('gill-spacing'),
    # col('gill-size'),
    # col('gill-color'),
    # col('stalk-shape'),
    # col('stalk-root'),
    # col('stalk-surface-above-ring'),
    # col('stalk-surface-below-ring'),
    # col('stalk-color-above-ring'),
    # col('stalk-color-below-ring'),
    # col('veil-color'),
    # col('ring-number'),
    # col('ring-type'),
    # col('spore-print-color'),
    # col('population'),
    # col('habitat')
    # )
    # Define the schema for the new row
    new_row_schema = StructType([
        StructField("class", StringType(), True),
        StructField("cap-shape", StringType(), True),
        StructField("cap-surface", StringType(), True),
        StructField("cap-color", StringType(), True),
        StructField("bruises", StringType(), True),
        StructField("odor", StringType(), True),
        StructField("gill-attachment", StringType(), True),
        StructField("gill-spacing", StringType(), True),
        StructField("gill-size", StringType(), True),
        StructField("gill-color", StringType(), True),
        StructField("stalk-shape", StringType(), True),
        StructField("stalk-root", StringType(), True),
        StructField("stalk-surface-above-ring", StringType(), True),
        StructField("stalk-surface-below-ring", StringType(), True),
        StructField("stalk-color-above-ring", StringType(), True),
        StructField("stalk-color-below-ring", StringType(), True),
        StructField("veil-color", StringType(), True),
        StructField("ring-number", StringType(), True),
        StructField("ring-type", StringType(), True),
        StructField("spore-print-color", StringType(), True),
        StructField("population", StringType(), True),
        StructField("habitat", StringType(), True)
    ])

    # Create a new DataFrame with a single row
    new_row_df = spark.createDataFrame(
        [(predict_data.get(key) or "unknown") for key in new_row_schema.fieldNames()],
        schema=new_row_schema
    )

    # transform data
    first_row_df = df_original.limit(1)

    # df_original = first_row_df.select(df_reordered.columns)
    # merged_df = df_original.union(df_reordered)
    # merged_df.show()
    merged_df = first_row_df.union(new_row_df)

    df_encoded = model.transform(merged_df)
    print("Number of rows: "+ str(df_encoded.count()))

    # make prediction
    lr_prediction = lr_model.transform(df_encoded)


    # collect predictions into a list
    predictions = [row['prediction'] for row in lr_prediction.select('prediction').collect()]

    # format result as JSON
    json_result = {'predictions': predictions}
    return jsonify(json_result)
