# Spring 2023 CPSC 531-03 Final Project Implementaion
### Member: Justin Kim


## Project Functionality
1. 3 Parts:
   - Predicting Mushroom Type ( Poisonous or Edible ) Python Application using mushroom.csv from local storage
   - Predicting Mushroom Type ( Poisonous or Edible ) Python Application using Mongodb database collection
   - Mushroom Prediction Web Application using Flask, HTML, CSS


### Mushroom Prediction Python Application using Mongodb
   - Connects to Mongodb using Pymongo server and client to access cpsc531_FinalProject database mushroom collection
   - Creates sparkSession called 'ml-mushroomType'
   - Converts Mongodb mushroom collection to Spark Dataframe using sparkSession createDataFrame method
