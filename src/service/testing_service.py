from sklearn import metrics


def test_model(model, x_test, y_test):
    y_predicted = model.predict(x_test)

    cm = metrics.confusion_matrix(y_test, y_predicted)
    precision = metrics.precision_score(y_test, y_predicted, average=None)
    recall = metrics.recall_score(y_test, y_predicted, average=None)
    f1 = metrics.f1_score(y_test, y_predicted, average=None)

    print(f"f1 Score: {f1}\nPrecision: {precision}\nRecall: {recall}")
    print(cm)

    output = f"f1 Score: {f1}, Precision: {precision}, Recall: {recall}, Confusion Matrix: {cm}"

    return output
