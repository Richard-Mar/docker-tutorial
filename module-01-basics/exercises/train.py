"""
Exercise 2: sklearn 训练脚本
目标：把这个脚本打包进 Docker 镜像并运行
"""
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import json
import os

# 加载数据
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# 训练
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 评估
acc = accuracy_score(y_test, clf.predict(X_test))
result = {"accuracy": round(acc, 4), "n_samples_train": len(X_train)}

print(f"[Train] Accuracy: {acc:.4f}")

# 如果挂载了 outputs 目录，就写入结果
output_dir = "/workspace/outputs"
if os.path.exists(output_dir):
    with open(f"{output_dir}/result.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"[Train] Result saved to {output_dir}/result.json")
