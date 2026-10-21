"""Test model hiện tại với đánh giá thực tế lộn xộn."""
from demo import SentimentPredictor

CAU_THUC_TE = [
    # (câu, nhãn kỳ vọng)
    ("sp ok nhưng ship hơi lâu", 0),
    ("hàg tốt lém, đóng gói cẩn thận, 5*", 1),
    ("ko đc như mô tả, thất vọng", 0),
    ("đẹp xuất sắc, sẽ mua lại 😍", 1),
    ("ship nhanh, sp y hình, ok", 1),
    ("hàng bị lỗi, shop k rep, chán", 0),
    ("mua lần 2 r, vẫn ưng như lần đầu", 1),
    ("kém chất lượng, k đáng tiền", 0),
    ("bt, k có j đặc biệt", 0),
    ("tuyệt vờiiii, 10 điểm", 1),
    ("giá rẻ, chất lg ổn", 1),
    ("giao chậm, hàng móp, bực mình", 0),
    ("sản phẩm tốt, nhưng giá hơi cao", 1),
    ("dở tệ, đừng mua mn ơi", 0),
    ("shop tư vấn nhiệt tình, ủng hộ lần sau", 1),
    ("hàng giả, lừa đảo, report", 0),
    ("okela, 5 sao ⭐⭐⭐⭐⭐", 1),
    ("thất vọng tràn trề 😡😡", 0),
    ("chất lượng tuyệt vời, đáng tiền", 1),
    ("k đẹp như hình, hơi thất vọng", 0),
]

predictor = SentimentPredictor("linear_svc")

print("=" * 70)
print(" TEST MODEL VỚI ĐÁNH GIÁ THỰC TẾ HỖN LOẠN")
print("=" * 70)

correct = 0
for text, expected in CAU_THUC_TE:
    r = predictor.predict(text)
    ok = "✅" if r["pred"] == expected else "❌"
    if r["pred"] == expected:
        correct += 1
    exp_label = "POS" if expected == 1 else "NEG"
    print(f"{ok} [{exp_label}] {text}")
    print(f"     → dự đoán: {r['label']}")
    print(f"     → sau xử lý: '{r['processed']}'")

print("\n" + "=" * 70)
print(f"ĐÚNG: {correct}/{len(CAU_THUC_TE)}  ({correct/len(CAU_THUC_TE)*100:.1f}%)")