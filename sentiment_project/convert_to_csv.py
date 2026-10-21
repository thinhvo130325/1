import pandas as pd

# ============================================================
# BƯỚC 3.1: Đọc file Excel
# ============================================================
df_train = pd.read_excel("data/Training_dataset.xlsx")
df_test = pd.read_excel("data/Testing_dataset.xlsx")

print("Kích thước ban đầu:")
print("  Train:", df_train.shape)
print("  Test: ", df_test.shape)

# ============================================================
# BƯỚC 3.2: Ghép train + test
# ============================================================
df = pd.concat([df_train, df_test], ignore_index=True)
print(f"\nSau khi ghép: {df.shape[0]} mẫu, {df.shape[1]} cột")
print("Cột hiện có:", df.columns.tolist())

# ============================================================
# BƯỚC 3.3: Đổi tên cột về chuẩn 'text' và 'label'
# ============================================================
# ⚠️ SỬA THEO KẾT QUẢ TỪ BƯỚC 2!
# Nếu cột đã là 'text' và 'label' rồi thì bỏ qua đoạn này.
RENAME_MAP = {
    # 'comment': 'text',      # nếu cột bình luận tên là 'comment'
    # 'review': 'text',       # nếu tên là 'review'
    # 'content': 'text',      # nếu tên là 'content'
    # 'sentiment': 'label',   # nếu cột nhãn tên là 'sentiment'
}
if RENAME_MAP:
    df = df.rename(columns=RENAME_MAP)

# ============================================================
# BƯỚC 3.4: Chuẩn hóa giá trị nhãn về 0/1
# ============================================================
# In ra để biết nhãn đang là gì
print("\nGiá trị nhãn hiện tại:", df['label'].unique())

# Nếu nhãn là chuỗi (Positive/Negative), map về số
if df['label'].dtype == 'object':
    df['label'] = df['label'].str.lower().map({
        'positive': 1, 'pos': 1, 'tích cực': 1, '1': 1,
        'negative': 0, 'neg': 0, 'tiêu cực': 0, '0': 0,
    })

# Nếu nhãn là -1/1, chuyển -1 thành 0
if set(df['label'].unique()) == {-1, 1}:
    df['label'] = df['label'].map({-1: 0, 1: 1})

# Ép kiểu về int
df['label'] = df['label'].astype(int)

print("Giá trị nhãn sau chuẩn hóa:", sorted(df['label'].unique()))

# ============================================================
# BƯỚC 3.5: Làm sạch dữ liệu
# ============================================================
# Loại bỏ dòng thiếu comment hoặc label
truoc = len(df)
df = df.dropna(subset=['text', 'label'])
print(f"\nĐã loại {truoc - len(df)} dòng có giá trị thiếu")

# Loại bỏ comment rỗng (chỉ có khoảng trắng)
truoc = len(df)
df = df[df['text'].astype(str).str.strip() != '']
print(f"Đã loại {truoc - len(df)} dòng có comment rỗng")

# Loại bỏ trùng lặp
truoc = len(df)
df = df.drop_duplicates(subset=['text']).reset_index(drop=True)
print(f"Đã loại {truoc - len(df)} dòng trùng lặp")

# ============================================================
# BƯỚC 3.6: Chỉ giữ 2 cột cần thiết
# ============================================================
df = df[['text', 'label']]

# ============================================================
# BƯỚC 3.7: Kiểm tra mất cân bằng
# ============================================================
print("\n" + "=" * 50)
print("THỐNG KÊ CUỐI CÙNG")
print("=" * 50)
print(f"Tổng số mẫu: {len(df)}")
print("\nPhân bố nhãn:")
counts = df['label'].value_counts().sort_index()
for label, count in counts.items():
    ten = "Positive" if label == 1 else "Negative"
    print(f"  {label} ({ten}): {count:>6} mẫu  ({count/len(df)*100:.1f}%)")

# Cảnh báo nếu < 2000 mẫu
if len(df) < 2000:
    print(f"\n⚠️  CẢNH BÁO: Chỉ có {len(df)} mẫu, ít hơn yêu cầu 2.000!")
    print("   Cần bổ sung dữ liệu từ nguồn khác (SA-VLSP2016, AIVIVN, ...)")
else:
    print(f"\n✅ Đủ {len(df)} mẫu (>= 2.000)")

# ============================================================
# BƯỚC 3.8: Lưu file CSV
# ============================================================
output_path = "data/positive_negative_feedback.csv"
df.to_csv(output_path, index=False, encoding='utf-8')
print(f"\n✅ Đã lưu: {output_path}")
print(f"   Kích thước file: {df.shape} (dòng, cột)")