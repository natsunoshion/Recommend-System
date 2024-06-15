from model import *

mySVD = SVDModel()

# load the dataset
mySVD.loadTrainSet()
m1 = get_process_memory()

# train the model
begin = time.time()
mySVD.train()
end = time.time()
duration = end - begin
print("Train time: ", "%.6f" % duration, "seconds")
m2 = get_process_memory()

# evaluate the model
begin = time.time()
mySVD.evaluate()
end = time.time()
duration = end - begin
print("Evaluation time: ", "%.6f" % duration, "seconds")
m3 = get_process_memory()


# test the model
begin = time.time()
mySVD.predictOnTestDataset()
end = time.time()
duration = end - begin
print("Prediction time: ", "%.6f" % duration, "seconds")
m4 = get_process_memory()


# get the maximum memory that a process used
memory_usage = max(m1, m2, m3, m4)

print(
    f"Model training Memory usage(it indicates the maximum memory that a process used): {memory_usage} MB"
)
