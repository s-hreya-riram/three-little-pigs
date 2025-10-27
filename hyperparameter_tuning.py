from pyspark.sql import SparkSession
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler

# 1. Initialize Spark
spark = SparkSession.builder.appName("InsuranceHyperTune").getOrCreate()

# 2. Load data (or from locally done cleaning?)
df = spark.read.csv("gs://bucket-three-little-pigs-476102/insurance.csv", header=True, inferSchema=True)

df = df.dropna()

# 3. Assemble features
assembler = VectorAssembler(
    inputCols=[col for col in df.columns if col not in ('charges',)],
    outputCol="features"
)
df = assembler.transform(df).select("features", "charges")

# 4. Example model: Gradient Boosted Trees
gbt = GBTClassifier(labelCol="charges", featuresCol="features")

# 5. Define hyperparameter grid
paramGrid = (ParamGridBuilder()
             .addGrid(gbt.maxDepth, [3, 5, 7])
             .addGrid(gbt.maxIter, [10, 20])
             .build())

# 6. Cross-validation
evaluator = BinaryClassificationEvaluator(labelCol="charges")
cv = CrossValidator(estimator=gbt,
                    estimatorParamMaps=paramGrid,
                    evaluator=evaluator,
                    numFolds=3)

cvModel = cv.fit(df)

# 7. Save best model and metrics
bestModel = cvModel.bestModel
bestModel.save("gs://bucket-three-little-pigs-476102/output/best_model")
print("Best model params:", bestModel.extractParamMap())

spark.stop()

# to submit job:
# !gcloud dataproc jobs submit pyspark gs://bucket-three-little-pigs-476102/scripts/hyperparameter_tuning.py \
#  --cluster=cluster-insurance-pigs \
#  --region=asia-southeast1 \
#  --project=three-little-pigs-476102

# check bucket for outcome

# terminate cluster to prevent add monies
# gcloud dataproc clusters delete cluster-insurance-pigs \
#  --region=asia-southeast1
