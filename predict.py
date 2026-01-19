import pickle

with open("model.bin", "rb") as f_in:
    pipeline = pickle.load(f_in)


def predict(sensor_data):
    result = pipeline.predict_proba([sensor_data])[0, 1]
    return float(result)


def lambda_handler(event, context):
    sensor_data = event["sensor_data"]
    pass_prob = predict(sensor_data)
    threshold = 0.28  # Best threshold value
    outcome = "fail" if pass_prob >= threshold else "pass"
    return {"predicted proba": pass_prob, "predicted test outcome": outcome}
