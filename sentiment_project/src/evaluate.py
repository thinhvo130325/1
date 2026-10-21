"""
Script đánh giá: vẽ confusion matrix + trích top features.
Chạy SAU khi đã có models trong thư mục models/.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from imblearn.over_sampling import SMOTE

from config import DATA_PATH, MODEL_DIR, TFIDF_PARAMS, TEST_SIZE, RANDOM_STATE
from preprocess import preprocess_pipeline

# ============================================================
# 1. VẼ CONFUSION MATRIX CHO CẢ 3 MÔ HÌNH (CẠNH NHAU)
# ============================================================
def ve_confusion_matrix():
    # Đọc dữ liệu
    df = pd.read_csv(DATA_PATH)
    if "comment" not in df.columns and "text" in df.columns:
        df = df.rename(columns={"text": "comment"})
    df = df.dropna(subset=["comment", "label"])
    df["processed"] = df["comment"].astype(str).apply(preprocess_pipeline)

    X_text = df["processed"].astype(str).to_numpy(dtype=object)
    y = df["label"].astype(int).to_numpy()

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)
    X_train_res, y_train_res = SMOTE(random_state=RANDOM_STATE).fit_resample(X_train, y_train)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Linear SVC": LinearSVC(random_state=RANDOM_STATE),
        "Multinomial NB": MultinomialNB(),
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (name, model) in zip(axes, models.items()):
        model.fit(X_train_res, y_train_res)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        cm = ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred,
            display_labels=["Negative", "Positive"],
            cmap="Blues", ax=ax, colorbar=False,
            values_format="d",
        )
        ax.set_title(f"{name}\nAccuracy: {acc:.4f}", fontsize=11, fontweight="bold")

    plt.tight_layout()
    out = "confusion_matrices.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"✅ Đã lưu {out}")
    plt.show()


# ============================================================
# 2. TRÍCH TOP FEATURES (TỪ/ĐẶC TRƯNG TRỌNG SỐ CAO NHẤT)
# ============================================================
def trich_top_features(top_n=15):
    vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
    feature_names = vectorizer.get_feature_names_out()

    # Hàm chung cho LR và SVC (dùng coef_)
    def in_top(model_path, model_name):
        model = joblib.load(model_path)
        if not hasattr(model, "coef_"):
            print(f"⚠️  {model_name} không có coef_, bỏ qua.")
            return
        coef = model.coef_[0]
        top_pos_idx = np.argsort(coef)[-top_n:][::-1]
        top_neg_idx = np.argsort(coef)[:top_n]

        print(f"\n{'='*60}")
        print(f"TOP {top_n} ĐẶC TRƯNG — {model_name}")
        print(f"{'='*60}")

        print(f"\n🟢 TÍCH CỰC (trọng số dương cao):")
        for i in top_pos_idx:
            print(f"   {feature_names[i]:30s}  {coef[i]:+.4f}")

        print(f"\n🔴 TIÊU CỰC (trọng số âm):")
        for i in top_neg_idx:
            print(f"   {feature_names[i]:30s}  {coef[i]:+.4f}")

    in_top(os.path.join(MODEL_DIR, "linear_svc.pkl"), "Linear SVC")
    in_top(os.path.join(MODEL_DIR, "logistic_regression.pkl"), "Logistic Regression")


# ============================================================
# 3. SO SÁNH TRƯỚC VÀ SAU SMOTE (ĐIỂM CỘNG)
# ============================================================
def so_sanh_smote():
    df = pd.read_csv(DATA_PATH)
    if "comment" not in df.columns and "text" in df.columns:
        df = df.rename(columns={"text": "comment"})
    df = df.dropna(subset=["comment", "label"])
    df["processed"] = df["comment"].astype(str).apply(preprocess_pipeline)

    X_text = df["processed"].astype(str).to_numpy(dtype=object)
    y = df["label"].astype(int).to_numpy()

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    print("\n" + "="*60)
    print("SO SÁNH TRƯỚC VÀ SAU SMOTE (Linear SVC)")
    print("="*60)

    # Không SMOTE
    model_no_smote = LinearSVC(random_state=RANDOM_STATE)
    model_no_smote.fit(X_train, y_train)
    y_pred_no = model_no_smote.predict(X_test)
    report_no = classification_report(y_test, y_pred_no, target_names=["Neg", "Pos"], digits=4)

    # Có SMOTE
    X_res, y_res = SMOTE(random_state=RANDOM_STATE).fit_resample(X_train, y_train)
    model_smote = LinearSVC(random_state=RANDOM_STATE)
    model_smote.fit(X_res, y_res)
    y_pred_sm = model_smote.predict(X_test)
    report_sm = classification_report(y_test, y_pred_sm, target_names=["Neg", "Pos"], digits=4)

    print("\n❌ KHÔNG SMOTE:")
    print(report_no)
    print(f"\n✅ CÓ SMOTE:")
    print(report_sm)


if __name__ == "__main__":
    print("Chọn chức năng:")
    print("  1. Vẽ confusion matrix")
    print("  2. Trích top features")
    print("  3. So sánh trước/sau SMOTE")
    print("  4. Chạy tất cả")
    chon = input("\nNhập lựa chọn (1-4): ").strip()

    if chon in ("1", "4"):
        ve_confusion_matrix()
    if chon in ("2", "4"):
        trich_top_features()
    if chon in ("3", "4"):
        so_sanh_smote()