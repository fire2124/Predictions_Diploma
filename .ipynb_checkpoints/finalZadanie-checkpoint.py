#!/usr/bin/env python2.7
#spustenie na serveri s pythonom 2.7


#Importy
import sys
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql import SQLContext
from pyspark.sql.functions import when
from pyspark.mllib.random import RandomRDDs
from pyspark.ml.classification import LinearSVC
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import NaiveBayes, NaiveBayesModel
from pyspark.ml.classification import DecisionTreeClassifier, DecisionTreeClassificationModel
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.util import MLUtils


sc = SparkContext(appName="example23")
print(u'Python version ' + sys.version)
print(u'Spark version ' + sc.version)
spark = SparkSession.builder.appName("Zadanie").getOrCreate()
sqlContext = SQLContext(sc)

# nacitanie dat
data=spark.read.format('csv').options(header='true', inferSchema='true').load('C:/Users/FIlip/Documents/TSVD/Zadanie/dataset/Accidents.csv')
data1=spark.read.format('csv').options(header='true', inferSchema='true').load('C:/Users/FIlip/Documents/TSVD/Zadanie/dataset/Vehicles.csv')
data2=spark.read.format('csv').options(header='true', inferSchema='true').load('C:/Users/FIlip/Documents/TSVD/Zadanie/dataset/Casualties.csv')

data1=data1.withColumnRenamed("Accident_Index","ID")
data2=data2.withColumnRenamed("Accident_Index","IDE")

#spojenie dat a vymazanie potom tych duplikatych ID stlpcov
merge = data.join(data1, data.Accident_Index == data1.ID)
merge.drop("ID")
full_merge = merge.join(data2, merge.Accident_Index == data2.IDE)
full_merge=full_merge.drop("IDE")
full_merge=full_merge.drop("ID")

# čiže ci cieľovy atribut je Accident_Severity a ma 3 hodnoty : 1, 2, 3 - jedna je smrtelna 2 je že važna ale nezomrel a 3 je že ľahka... tak som 3 replacol na 2 aby bolo len smrtelna
# a niesmrtelna
newsdf = full_merge.withColumn("Accident_Severity", when(full_merge["Accident_Severity"] == 3, 2).otherwise(full_merge["Accident_Severity"]))


#cize sampling  som si vytiahol koľko je počet mrtvych a koľko počet čo prežili nehodu a podľa toho som v tokom pomere zmenšil tie data na 5%
newsdf.registerTempTable("TempTable")
mrtvy = sqlContext.sql('SELECT * FROM TempTable WHERE Accident_Severity = 1')
mrtvy_pocet = mrtvy.count()
zivy = sqlContext.sql('SELECT * FROM TempTable WHERE Accident_Severity = 2')
zivy_pocet = zivy.count()
vzorka_mrtvy = mrtvy.sampleBy("Accident_Severity", fractions = {1: 1}, seed = 0)
vzorka_zivy = zivy.sampleBy("Accident_Severity", fractions = {2: 0.02}, seed = 0)
#tu ich nazad spojim a s tym datasetom treba pracovať ďalej
vzorka_cela = vzorka_mrtvy.union(vzorka_zivy)

print(vzorka_mrtvy.count())
print(vzorka_zivy.count())
print(vzorka_cela.count())

#Statistiky 
Accident_Severity = vzorka_cela.describe(["Accident_Severity"])
Accident_Severity.collect()
Number_of_Vehicles = vzorka_cela.describe(["Number_of_Vehicles"])
Number_of_Vehicles.collect()
Number_of_Casualties = vzorka_cela.describe(["Number_of_Casualties"])
Number_of_Casualties.collect()
Road_Class = vzorka_cela.describe(["1st_Road_Class"])
Road_Class.collect()
Speed_limit = vzorka_cela.describe(["Speed_limit"])
Speed_limit.collect()
Junction_Detail = vzorka_cela.describe(["Junction_Detail"])
Junction_Detail.collect()
Junction_Control = vzorka_cela.describe(["Junction_Control"])
Junction_Control.collect()
nd_Road_Class = vzorka_cela.describe(["2nd_Road_Class"])
nd_Road_Class.collect()
Urban_or_Rural_Area = vzorka_cela.describe(["Urban_or_Rural_Area"])
Urban_or_Rural_Area.collect()
Did_Police_Officer_Attend_Scene_of_Accident = vzorka_cela.describe(["Did_Police_Officer_Attend_Scene_of_Accident"])
Did_Police_Officer_Attend_Scene_of_Accident.collect()
Vehicle_Manoeuvre = vzorka_cela.describe(["Vehicle_Manoeuvre"])
Vehicle_Manoeuvre.collect()
Junction_Location = vzorka_cela.describe(["Junction_Location"])
Junction_Location.collect()
Skidding_and_Overturning = vzorka_cela.describe(["Skidding_and_Overturning"])
Skidding_and_Overturning.collect()
Vehicle_Leaving_Carriageway = vzorka_cela.describe(["Vehicle_Leaving_Carriageway"])
Vehicle_Leaving_Carriageway.collect()
Hit_Object_off_Carriageway = vzorka_cela.describe(["Hit_Object_off_Carriageway"])
Hit_Object_off_Carriageway.collect()
Casualty_Reference = vzorka_cela.describe(["Casualty_Reference"])
Casualty_Reference.collect()
Casualty_Severity = vzorka_cela.describe(["Casualty_Severity"])
Casualty_Severity.collect()

#nahradenie neznamych hodnot atributov
atributes = ["Number_of_Vehicles",
			 "Number_of_Casualties",
			 "Junction_Detail",
			 "Junction_Control",
			 "Did_Police_Officer_Attend_Scene_of_Accident",
			 "Junction_Location",
			 "Skidding_and_Overturning",
			 "Hit_Object_off_Carriageway",
			 "Accident_Severity"]

vzorka_cela = vzorka_cela.select(atributes)

names2 = vzorka_cela.schema.names

for name in names2:
	type_counts = vzorka_cela.groupBy(name).count()
	type_counts = type_counts.orderBy(["count", name], ascending=[0, 1])
	moj_list = type_counts.select(name).collect()
	najcastejsia_hodnota = moj_list[0][0]
	vzorka_cela = vzorka_cela.withColumn(name, when(vzorka_cela[name] == -1,  najcastejsia_hodnota).otherwise(vzorka_cela[name]))

#kontrola ci nahradilo
for name in names2:
	x = vzorka_cela.filter(vzorka_cela[name] == -1).count()
	print(x)
#Nakoniec
vzorka_cela = vzorka_cela.withColumn("Junction_Control", when(vzorka_cela["Junction_Control"] == -1, 2).otherwise(vzorka_cela["Junction_Control"]))


# #najprv si dropnem tie atributy ktore tam nechcem mat
vzorka_cela = vzorka_cela.drop(
					 "Local_Authority_(Highway)",
 					 "LSOA_of_Accident_Location",
					 "Time",
					 "Date",
					 "Accident_Index")

#do premennej names si ulozim mena atributov
names = vzorka_cela.schema.names
print(names)

#print(names.count)
#vytvorim si prazdny list correlations a do neho vo for cykle hodim korelacie kazdeho atributu s #atributom accident_severity, ale dajak to nefujguje zatial :D

correlations = []
for name in names:
    correlations.extend([vzorka_cela.stat.corr('Accident_Severity',name)])

print(correlations)                       

names = vzorka_cela.schema.names
t = zip(names,correlations)
print(t)
print("-------------------")
tt = spark.createDataFrame(t)
tt.show()

tt.registerTempTable("TempTable")
atributy_table = sqlContext.sql('SELECT * FROM TempTable WHERE _2 > 0.05 OR _2 <-0.05')
atributy = atributy_table.select("_2")
atributy_table.show()

cast_vzorky= vzorka_cela.drop("Location_Easting_OSGR",
					 "Location_Northing_OSGR",
					"Longitude",
					 "Latitude",
					 "Local_Authority_(District)",
					 "1st_Road_Class",
					 "1st_Road_Number",
					 "Road_Type",
					 "Speed_limit",
					 "2nd_Road_Class",
					 "2nd_Road_Number",
					"Pedestrian_Crossing-Human_Control",
					"Pedestrian_Crossing-Physical_Facilities",
					 "Special_Conditions_at_Site",
					 "Carriageway_Hazards",
					 "Urban_or_Rural_Area",
					 "Vehicle_Reference",
					 "Towing_and_Articulation",
					 "Vehicle_Manoeuvre",
					"Vehicle_Location-Restricted_Lane",
					 "Vehicle_Leaving_Carriageway",
					 "Was_Vehicle_Left_Hand_Drive?",
					 "Journey_Purpose_of_Driver",
					 "Propulsion_Code",
					 "Driver_IMD_Decile",
					 "Driver_Home_Area_Type",
					"Vehicle_Reference",
					 "Casualty_Reference",
					 "Police_Force",
                     "Day_of_Week",
                     "Light_Conditions",
                     "Weather_Conditions",
                     "Road_Surface_Conditions",
                     "Vehicle_Type",
                     "Hit_Object_in_Carriageway",
                      "1st_Point_of_Impact",
                     "Sex_of_Driver",
                     "Age_of_Driver",
                    "Age_Band_of_Driver",
                    "Engine_Capacity_(CC)",
                    "Age_of_Vehicle",
                    "Casualty_Class",
                    "Sex_of_Casualty",
                    "Age_of_Casualty",
                    "Age_Band_of_Casualty",
                    "Casualty_Severity",
                    "Pedestrian_Location",
                    "Pedestrian_Movement",
                    "Car_Passenger",
                    "Bus_or_Coach_Passenger",
                    "Pedestrian_Road_Maintenance_Worker",
                    "Casualty_Type",
                    "Casualty_Home_Area_Type")
names = cast_vzorky.schema.names
print(names)

#upravim hodnoty niektorych atributov
cast_vzorky = cast_vzorky.withColumn("Hit_Object_off_Carriageway", when(
		(cast_vzorky["Hit_Object_off_Carriageway"] == -1) |
		(cast_vzorky["Hit_Object_off_Carriageway"] == 1) |
		(cast_vzorky["Hit_Object_off_Carriageway"] == 2) |
		(cast_vzorky["Hit_Object_off_Carriageway"] == 3) |
		(cast_vzorky["Hit_Object_off_Carriageway"] == 4) |
		(cast_vzorky["Hit_Object_off_Carriageway"] == 5) |
		(cast_vzorky["Hit_Object_off_Carriageway"] == 6) |
		(cast_vzorky["Hit_Object_off_Carriageway"] == 7) |
		(cast_vzorky["Hit_Object_off_Carriageway"] == 8) |
		(cast_vzorky["Hit_Object_off_Carriageway"] == 9) |
		(cast_vzorky["Hit_Object_off_Carriageway"] == 10) |
		(cast_vzorky["Hit_Object_off_Carriageway"] == 11),1).otherwise(cast_vzorky["Hit_Object_off_Carriageway"]))
   
cast_vzorky = cast_vzorky.withColumn("Junction_Detail", when(
		(cast_vzorky["Junction_Detail"] == 1) |
		(cast_vzorky["Junction_Detail"] == 2) |
		(cast_vzorky["Junction_Detail"] == 3) |
		(cast_vzorky["Junction_Detail"] == 4) |
		(cast_vzorky["Junction_Detail"] == 5) |
		(cast_vzorky["Junction_Detail"] == 6) |
		(cast_vzorky["Junction_Detail"] == 7) |
		(cast_vzorky["Junction_Detail"] == 8) |
		(cast_vzorky["Junction_Detail"] == 9), 1).otherwise(cast_vzorky["Junction_Detail"]))
    
cast_vzorky = cast_vzorky.withColumn("Junction_Location", when(
		(cast_vzorky["Junction_Location"] == 1) |
		(cast_vzorky["Junction_Location"] == 2) |
		(cast_vzorky["Junction_Location"] == 3) |
		(cast_vzorky["Junction_Location"] == 4) |
		(cast_vzorky["Junction_Location"] == 5) |
		(cast_vzorky["Junction_Location"] == 6) |
		(cast_vzorky["Junction_Location"] == 7) |
		(cast_vzorky["Junction_Location"] == 8), 1).otherwise(cast_vzorky["Junction_Location"]))
    
cast_vzorky = cast_vzorky.withColumn("Skidding_and_Overturning", when(
		(cast_vzorky["Skidding_and_Overturning"] == 1) |
		(cast_vzorky["Skidding_and_Overturning"] == 2) |
		(cast_vzorky["Skidding_and_Overturning"] == 3) |
		(cast_vzorky["Skidding_and_Overturning"] == 4) |
		(cast_vzorky["Skidding_and_Overturning"] == 5), 1).otherwise(cast_vzorky["Skidding_and_Overturning"]))



#zmena na accident severity na binarny atribut
SVM_df = cast_vzorky.withColumn("Accident_Severity", when(cast_vzorky["Accident_Severity"] == 1, 0).otherwise(cast_vzorky["Accident_Severity"]))
SVM_df = SVM_df.withColumn("Accident_Severity", when(SVM_df["Accident_Severity"] == 2, 1).otherwise(SVM_df["Accident_Severity"]))

SVM_df.describe("Accident_Severity").collect()
SVM_df = VectorAssembler(inputCols=["Number_of_Vehicles", "Number_of_Casualties", "Junction_Detail", 
                                             "Junction_Control", "Did_Police_Officer_Attend_Scene_of_Accident", 
                                             "Junction_Location","Skidding_and_Overturning","Hit_Object_off_Carriageway"],
        outputCol="features").transform(SVM_df) 
#Rozdelenie dat na trenovaciu a testovaciu mnozinu
training_data, test_data = SVM_df.randomSplit([0.6, 0.4], seed=123)
SVM_df.count()


#Modelovanie
print "---------------------------------------------------------------------"
print "-----------------------------Modelovanie-----------------------------"
print "---------------------------------------------------------------------"

#Decision tree classifier
print "-------------------------------------------------"
print "---------------DESICION TREE---------------"
print "-------------------------------------------------"

tree_classifier = DecisionTreeClassifier(featuresCol="features",labelCol="Accident_Severity",impurity="entropy",maxDepth=10, maxBins=100) 
tree_model = tree_classifier.fit(training_data)
predictions = tree_model.transform(test_data)
#print(tree_model.toDebugString)
test_error = predictions.filter(predictions["prediction"] != predictions["Accident_Severity"]).count() / float(test_data.count())
print "Testing error: {0:.4f}".format(test_error)
# Select example rows to display.
predictions.select("prediction", "Accident_Severity", "features").show(5)
#Model rozhodovacie stromu
print(tree_model.toDebugString)
#vyhodnotenie decision tree
evaluatorMulti = MulticlassClassificationEvaluator(labelCol="Accident_Severity", predictionCol="prediction")
evaluator = BinaryClassificationEvaluator(labelCol="Accident_Severity", rawPredictionCol="prediction", metricName='areaUnderROC')
acc = evaluatorMulti.evaluate(predictions, {evaluatorMulti.metricName: "accuracy"})
f1 = evaluatorMulti.evaluate(predictions, {evaluatorMulti.metricName: "f1"})
Precision = evaluatorMulti.evaluate(predictions, {evaluatorMulti.metricName: "weightedPrecision"})
Recall = evaluatorMulti.evaluate(predictions, {evaluatorMulti.metricName: "weightedRecall"})
auc = evaluator.evaluate(predictions)
print('Accuracy score: ',acc)
print('f1: ',f1)
print('Precision: ',Precision)
print('Recall: ',Recall)
print('Auc: ',auc)
#kontingencna tabulka
cf = predictions.crosstab("prediction","Accident_Severity")
cf.show()



print "-------------------------------------------------"
print "---------------LogisticRegression---------------"
print "-------------------------------------------------"
#Logisticka regresia
lr = LogisticRegression(featuresCol = 'features', labelCol = 'Accident_Severity', maxIter=10)
lrModel = lr.fit(training_data)
predictions = lrModel.transform(test_data)
predictions.select("prediction", "Accident_Severity", "features").show(10)
print(predictions)
#kontingencna tabulka Logisticka regresia
cf = predictions.crosstab("prediction","Accident_Severity")
cf.show()
#vyhodnotenie Logisticka regresia
evaluatorMulti = MulticlassClassificationEvaluator(labelCol="Accident_Severity", predictionCol="prediction")
evaluator = BinaryClassificationEvaluator(labelCol="Accident_Severity", rawPredictionCol="prediction", metricName='areaUnderROC')
acc = evaluatorMulti.evaluate(predictions, {evaluatorMulti.metricName: "accuracy"})
f1 = evaluatorMulti.evaluate(predictions, {evaluatorMulti.metricName: "f1"})
Precision = evaluatorMulti.evaluate(predictions, {evaluatorMulti.metricName: "weightedPrecision"})
Recall = evaluatorMulti.evaluate(predictions, {evaluatorMulti.metricName: "weightedRecall"})
auc = evaluator.evaluate(predictions)
print('Accuracy score: ',acc)
print('f1: ',f1)
print('Precision: ',Precision)
print('Recall: ',Recall)
print('Auc: ',auc)


#SVM
print "-------------------------------------------------"
print "-----------------------SVM-----------------------"
print "-------------------------------------------------"
svm_classifier = LinearSVC(featuresCol="features",labelCol="Accident_Severity")                  
svm_model = svm_classifier.fit(training_data)
predictions = svm_model.transform(test_data)
test_error = predictions.filter(predictions["prediction"] != predictions["Accident_Severity"]).count() / float(test_data.count())
print "Testing error: {0:.4f}".format(test_error)
#kontingencna tabulka SVM
cf = predictions.crosstab("prediction","Accident_Severity")
cf.show()
#vyhodnotenie SVM
evaluatorMulti = MulticlassClassificationEvaluator(labelCol="Accident_Severity", predictionCol="prediction")
evaluator = BinaryClassificationEvaluator(labelCol="Accident_Severity", rawPredictionCol="prediction", metricName='areaUnderROC')
acc = evaluatorMulti.evaluate(predictions, {evaluatorMulti.metricName: "accuracy"})
f1 = evaluatorMulti.evaluate(predictions, {evaluatorMulti.metricName: "f1"})
Precision = evaluatorMulti.evaluate(predictions, {evaluatorMulti.metricName: "weightedPrecision"})
Recall = evaluatorMulti.evaluate(predictions, {evaluatorMulti.metricName: "weightedRecall"})
auc = evaluator.evaluate(predictions)
print('Accuracy score: ',acc)
print('f1: ',f1)
print('Precision: ',Precision)
print('Recall: ',Recall)
print('Auc: ',auc)



# bayes 
print "-------------------------------------------------"
print "--------------------NaiveBayes-------------------"
print "-------------------------------------------------"
nb = NaiveBayes(smoothing=1.0, modelType="multinomial",featuresCol="features", labelCol="Accident_Severity")
 # train the model
model = nb.fit(training_data)
predictions = model.transform(test_data)
evaluator = MulticlassClassificationEvaluator(labelCol="Accident_Severity", predictionCol="prediction",
                                              metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test set accuracy = " + str(accuracy))
#kontingencna tabulka bayes
cf = predictions.crosstab("prediction","Accident_Severity")
cf.show()
#vyhodnotenie bayes
evaluatorMulti = MulticlassClassificationEvaluator(labelCol="Accident_Severity", predictionCol="prediction")
evaluator = BinaryClassificationEvaluator(labelCol="Accident_Severity", rawPredictionCol="prediction", metricName='areaUnderROC')
acc = evaluatorMulti.evaluate(predictions, {evaluatorMulti.metricName: "accuracy"})
f1 = evaluatorMulti.evaluate(predictions, {evaluatorMulti.metricName: "f1"})
Precision = evaluatorMulti.evaluate(predictions, {evaluatorMulti.metricName: "weightedPrecision"})
Recall = evaluatorMulti.evaluate(predictions, {evaluatorMulti.metricName: "weightedRecall"})
auc = evaluator.evaluate(predictions)
print('Accuracy score: ',acc)
print('f1: ',f1)
print('Precision: ',Precision)
print('Recall: ',Recall)
print('Auc: ',auc)



#Random Forest
print "-------------------------------------------------"
print "------------------Random Forest------------------"
print "-------------------------------------------------"
# Train a RandomForest model.
rf = RandomForestClassifier(labelCol="Accident_Severity", featuresCol="features",impurity="entropy", numTrees=10, maxBins=100)
# Train model.  This also runs the indexers.
model = rf.fit(training_data)
# Make predictions.
predictions = model.transform(test_data)
test_error = predictions.filter(predictions["prediction"] != predictions["Accident_Severity"]).count() / float(test_data.count())
print "Testing error: {0:.4f}".format(test_error)
print(model.toDebugString)
#kontingencna Random Forest
cf = predictions.crosstab("prediction","Accident_Severity")
cf.show()
#vyhodnotenie Random Forest
evaluatorMulti = MulticlassClassificationEvaluator(labelCol="Accident_Severity", predictionCol="prediction")
evaluator = BinaryClassificationEvaluator(labelCol="Accident_Severity", rawPredictionCol="prediction", metricName='areaUnderROC')
acc = evaluatorMulti.evaluate(predictions, {evaluatorMulti.metricName: "accuracy"})
f1 = evaluatorMulti.evaluate(predictions, {evaluatorMulti.metricName: "f1"})
Precision = evaluatorMulti.evaluate(predictions, {evaluatorMulti.metricName: "weightedPrecision"})
Recall = evaluatorMulti.evaluate(predictions, {evaluatorMulti.metricName: "weightedRecall"})
auc = evaluator.evaluate(predictions)
print('Accuracy score: ',acc)
print('f1: ',f1)
print('Precision: ',Precision)
print('Recall: ',Recall)
print('Auc: ',auc)



print "-------------------------------------------------"
print "---------------------KMeans----------------------"
print "-------------------------------------------------"
#Trains a k-means model.
kmeans = KMeans().setK(3).setSeed(1234)
model = kmeans.fit(training_data)
# Evaluate clustering by computing Within Set Sum of Squared Errors.
wssse = model.computeCost(training_data)
print("Within Set Sum of Squared Errors = " + str(wssse))
print("------------------------------------------------")
# Shows the result.
centers = model.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)    
    print("------------------------------------------------")
#detekovanie anomalii
for center in centers:
    for point in center:
        if (point > 5 or -5 > point):
               print "anomalia: {0:.15f}".format(point)