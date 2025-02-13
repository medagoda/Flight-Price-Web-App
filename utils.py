import pickle

def load_model():
    with open("flight_rf.pkl", "rb") as file:
        model = pickle.load(file)
    return model
