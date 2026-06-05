from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# 训练模型（容器启动时执行一次）
def load_model():
    iris = load_iris()
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(iris.data, iris.target)
    return clf, iris.target_names



# 全局加载一次
_model, _target_names = load_model()

def predict(features: list[float]) -> dict:
    pred = _model.predict([features])[0]
    return {
        "class_id": int(pred),
        "class_name": _target_names[pred]
    }
