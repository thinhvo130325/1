import os
import joblib
import numpy as np
from preprocess import preprocess_pipeline
from config import MODEL_DIR

class SentimentPredictor:
    def __init__(self, model_name="linear_svc"):
        self.vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
        self.model = joblib.load(os.path.join(MODEL_DIR, f"{model_name}.pkl"))
        self.model_name = model_name

    def predict(self, text: str, return_proba: bool = True):
        processed = preprocess_pipeline(text)
        X = self.vectorizer.transform([processed])
        pred = self.model.predict(X)[0]

        label = "TÍCH CỰC 😊" if pred == 1 else "TIÊU CỰC 😞"
        result = {"input": text, "processed": processed, "label": label, "pred": int(pred)}

        if return_proba and hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)[0]
            result["confidence"] = float(np.max(proba))
            result["proba"] = proba.tolist()
        elif hasattr(self.model, "decision_function"):
            score = self.model.decision_function(X)[0]
            # Chuyển score thành pseudo-probability qua sigmoid
            prob = 1 / (1 + np.exp(-score))
            result["confidence"] = float(prob if pred == 1 else 1 - prob)
            result["decision_score"] = float(score)

        return result

if __name__ == "__main__":
    print("=== DEMO PHÂN TÍCH CẢM XÚC BÌNH LUẬN TIẾNG VIỆT ===")
    print("(Gõ 'exit' để thoát)\n")

    predictor = SentimentPredictor(model_name="linear_svc")  # hoặc logistic_regression

    test_sentences = [
        "Sản phẩm rất tốt, giao hàng nhanh, mình rất hài lòng",
        "Hàng kém chất lượng, thất vọng quá, không nên mua",
        "Đóng gói cẩn thận, shop tư vấn nhiệt tình, sẽ ủng hộ lần sau",
    ]

    for s in test_sentences:
        res = predictor.predict(s)
        print(f"Câu: {res['input']}")
        print(f"  -> Dự đoán: {res['label']}  (độ tin cậy: {res.get('confidence', 'N/A'):.2%})\n")

    while True:
        user_input = input("Nhập câu bình luận: ").strip()
        if user_input.lower() in ("exit", "quit", ""):
            break
        res = predictor.predict(user_input)
        print(f"  -> {res['label']}  |  Đã xử lý: '{res['processed']}'\n")