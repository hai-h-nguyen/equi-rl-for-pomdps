import joblib

# my_dict = {}
# my_dict["_top"] = 4
# my_dict["_size"] = 67

# joblib.dump(my_dict, "dict.pt")

buffer_dict = joblib.load("test.pt")
print(buffer_dict.keys())
a = buffer_dict["buffer_dict"]["_top"]
print(a)