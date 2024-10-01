import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Định nghĩa tên cột
columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class']

# Tải tập dữ liệu
data = pd.read_csv("Program/python/iris/iris.data", header=None, names=columns)

print(data)

# Chuẩn bị dữ liệu
X = data.iloc[:, 0:4].values
y = data['class'].values


# Hàm tính độ chính xác
def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)


# Phương pháp Hold-out
def holdout_method(X, y, test_size=0.3, random_state=42):
    np.random.seed(random_state)

    # Tạo dictionary để lưu trữ indices cho mỗi class
    class_indices = {cls: np.where(y == cls)[0] for cls in np.unique(y)}

    X_train, X_test, y_train, y_test = [], [], [], []

    for cls, indices in class_indices.items():
        np.random.shuffle(indices)
        split_point = int(len(indices) * (1 - test_size))

        X_train.extend(X[indices[:split_point]])
        X_test.extend(X[indices[split_point:]])
        y_train.extend([cls] * split_point)
        y_test.extend([cls] * (len(indices) - split_point))

    X_train, X_test = np.array(X_train), np.array(X_test)
    y_train, y_test = np.array(y_train), np.array(y_test)

    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Tính độ chính xác cho từng class
    class_accuracies = {}
    for cls in np.unique(y):
        cls_indices = y_test == cls
        cls_correct = np.sum((y_test[cls_indices] == y_pred[cls_indices]))
        cls_total = np.sum(cls_indices)
        class_accuracies[cls] = cls_correct / cls_total

    # Tính độ chính xác tổng thể
    overall_accuracy = np.mean(y_test == y_pred) * 100

    return overall_accuracy, class_accuracies, y_test, y_pred


# Phương pháp K-fold Cross-validation
def k_fold_cross_validation(X, y, k=5):
    fold_size = len(X) // k
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    scores = []
    for i in range(k):
        start = i * fold_size
        end = (i + 1) * fold_size if i < k - 1 else len(X)
        test_indices = indices[start:end]
        train_indices = np.concatenate([indices[:start], indices[end:]])

        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        scores.append(accuracy)

    return np.mean(scores) * 100


# Phương pháp Leave-One-Out Cross-validation (LOOCV)
def loocv(X, y):
    n_samples = len(X)
    correct_predictions = 0

    for i in range(n_samples):
        # Lấy 1 mẫu làm test, còn lại làm train
        X_test = X[i:i + 1]
        y_test = y[i]
        X_train = np.concatenate([X[:i], X[i + 1:]], axis=0)
        y_train = np.concatenate([y[:i], y[i + 1:]])

        # Huấn luyện mô hình trên tập train
        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(X_train, y_train)

        # Dự đoán nhãn cho mẫu test
        y_pred = model.predict(X_test)

        # Kiểm tra dự đoán
        if y_pred == y_test:
            correct_predictions += 1

    # Tính độ chính xác
    accuracy = (correct_predictions / n_samples) * 100
    return accuracy


# Thực hiện các phương pháp đánh giá
overall_accuracy, class_accuracies, y_test, y_pred = holdout_method(X, y)
k_fold_accuracy = k_fold_cross_validation(X, y)
loocv_accuracy = loocv(X, y)

# In kết quả
print("\nHold-out Method:")
print(f"Độ chính xác tổng thể: {overall_accuracy:.2f}%")
print("Độ chính xác theo từng class:")
for cls, accuracy in class_accuracies.items():
    print(f"  {cls}: {accuracy:.2f}")

print("\nK-fold Cross-validation (k=5):")
print(f"Độ chính xác trung bình: {k_fold_accuracy:.2f}%")

print("\nLeave-One-Out Cross-validation (LOOCV):")
print(f"Độ chính xác: {loocv_accuracy:.2f}%")