from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, split, lower
from pyspark.sql.types import StringType
from pyspark.sql import functions as F
from pyspark.sql.functions import lower, split, col
from pyspark.ml.feature import HashingTF, IDF
from pyspark.ml import Pipeline
import sys

def is_float(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

if __name__ == "__main__":
  if len(sys.argv) != 3:
    print("Usage: top_taxis.py <input_file> <output_file>")
    sys.exit(1)
  input_file1 = sys.argv[1]
  input_file2 = sys.argv[2]

  spark = SparkSession.builder.appName("Assignment4_Ques2").getOrCreate()

  wikipedia_pages = spark.read.text(input_file1)

  words_df = wikipedia_pages.select(explode(split(lower(wikipedia_pages.value), "\s+")).alias("word"))

  words_df = words_df.withColumn("word", F.regexp_replace("word", "[^a-zA-Z]", ""))
  words_df = words_df.filter(words_df.word != "")

  word_counts = words_df.groupBy("word").count()

  sorted_word_counts = word_counts.orderBy(F.desc("count"))

  top_words = sorted_word_counts.limit(20000)

  top_words_array = top_words.select("word").rdd.flatMap(lambda x: x).collect()

  print(top_words_array)

  wiki_df = spark.read.option("header", "false").csv(input_file2)

  wiki_df.head()

  wiki_df = wiki_df.withColumnRenamed("_c0", "docID").withColumnRenamed("_c1", "text")

  wiki_df.head()

  words_df = wiki_df.select("docID", split(lower(wiki_df.text), "\s+").alias("words"))

  hashing_tf = HashingTF(inputCol="words", outputCol="raw_features", numFeatures=20000)
  tf_features = hashing_tf.transform(words_df)

  idf = IDF(inputCol="raw_features", outputCol="tf_idf_features")
  idf_model = idf.fit(tf_features)
  tf_idf_matrix = idf_model.transform(tf_features)

  tf_idf_matrix.select("docID", "tf_idf_features").show(truncate=False)

  spark.stop()
    