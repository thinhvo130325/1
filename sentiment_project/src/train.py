import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE

from config import DATA_PATH, MODEL_DIR, TFIDF_PARAMS, TEST_SIZE, RANDOM_STATE
from preprocess import preprocess_pipeline

os.makedirs(MODEL_DIR, exist_ok=True)

def load_and_prepare_data():
    """Đọc CSV, giả định có cột 'comment' và 'label' (0/1)."""
    df = pd.read_csv(DATA_PATH)
    # Đổi tên cột nếu cần (tùy dataset thực tế)
    if "comment" not in df.columns:
        # Thử các tên cột phổ biến khác
        for col in ["text", "review", "content"]:
            if col in df.columns:
                df = df.rename(columns={col: "comment"})
                break
    if "label" not in df.columns:
        for col in ["sentiment", "target", "rating_class"]:
            if col in df.columns:
                df = df.rename(columns={col: "label"})
                break

    df = df.dropna(subset=["comment", "label"])
    print(f"Tổng số mẫu: {len(df)}")
    print("Phân bố nhãn:\n", df["label"].value_counts())

    print("Đang tiền xử lý văn bản...")
    df["processed"] = df["comment"].astype(str).apply(preprocess_pipeline)
    return df

def train_all_models(df):
    # ✅ FIX: Chuyển tường minh về numpy array kiểu object/string
    # để tránh lỗi ArrowExtensionArray của pandas+pyarrow
    X_text = df["processed"].astype(str).to_numpy(dtype=object)
    y = df["label"].astype(int).to_numpy()

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # TF-IDF fit trên train, transform trên test
    vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    # Áp dụng SMOTE CHỈ trên tập train
    print("Đang áp dụng SMOTE trên tập train...")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"Train gốc: {X_train.shape}, sau SMOTE: {X_train_res.shape}")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Linear SVC": LinearSVC(random_state=RANDOM_STATE),
        "Multinomial NB": MultinomialNB(),
    }

    results = {}
    for name, model in models.items():
        print(f"\n===== Huấn luyện: {name} =====")
        model.fit(X_train_res, y_train_res)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=["Negative (0)", "Positive (1)"])
        cm = confusion_matrix(y_test, y_pred)

        results[name] = {"model": model, "accuracy": acc, "report": report, "cm": cm, "y_pred": y_pred, "y_test": y_test}
        print(f"Accuracy: {acc:.4f}")
        print(report)
        print("Confusion Matrix:\n", cm)

    # Lưu vectorizer và các model
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
    for name, res in results.items():
        safe_name = name.lower().replace(" ", "_")
        joblib.dump(res["model"], os.path.join(MODEL_DIR, f"{safe_name}.pkl"))
    print(f"\nĐã lưu vectorizer và models vào {MODEL_DIR}")

    return vectorizer, results

if __name__ == "__main__":
    df = load_and_prepare_data()
    vectorizer, results = train_all_models(df)

    # In bảng tổng hợp
    print("\n\n===== BẢNG TỔNG HỢP ACCURACY =====")
    for name, res in results.items():
        print(f"{name:25s}: {res['accuracy']:.4f}")