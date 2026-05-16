import pickle
with open('model_package.pkl', 'rb') as f:
    pkg = pickle.load(f)
    print("Keys in PKL:", pkg.keys())
    if 'model_name' in pkg:
        print("Model Name:", pkg['model_name'])
    else:
        print("Model Name NOT found in PKL!")
