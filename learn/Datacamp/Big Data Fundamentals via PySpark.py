### A course on big data via PySpark
# Big data is defined by its "three Vs"
# Volume (how much data?), variety (how many different data sources?) and velocity (how fast can we
# access the data?)

# Load an array into PySpark shell
arr = range(1, 100)
arraydata = sc.parallelize(arraydata)

# Load a local file into PySpark shell
lines = sc.textFile(file_path)

# Check the number of partitions in fileRDD
print("Number of partitions in fileRDD is", fileRDD.getNumPartitions())

# Create a fileRDD_part from file_path with 5 partitions
fileRDD_part = sc.textFile(file_path, minPartitions = 5)

# Check the number of partitions in fileRDD_part
print("Number of partitions in fileRDD_part is", fileRDD_part.getNumPartitions())