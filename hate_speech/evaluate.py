from sklearn.metrics import f1_score
def evaluate(y_true, y_pred):
    return f1_score(y_true, y_pred)