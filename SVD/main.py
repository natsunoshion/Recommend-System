from model import *

mySVD=SVDModel()

# load the dataset
mySVD.loadTrainSet()
m1=getProcessMemory()

# train the model
begin=time.time()
mySVD.train()
end=time.time()
duration = end - begin
print("Train time: ", "%.6f" % duration, "seconds")
m2=getProcessMemory()

# evaluate the model
begin=time.time()
mySVD.evaluate()
end=time.time()
duration = end - begin
print("Evaluation time: ", "%.6f" % duration, "seconds")
m3=getProcessMemory()


# test the model
begin=time.time()
mySVD.predictOnTestDataset()
end=time.time()
duration = end - begin
print("Prediction time: ", "%.6f" % duration, "seconds")
m4=getProcessMemory()


# get the maximum memory that a process used
memory_usage =max(m1,m2,m3,m4)

print(f"Model training Memory usage(it indicates the maximum memory that a process used): {memory_usage} MB")
