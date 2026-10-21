import pandas as pd

# Đọc 2 file Excel
df_train = pd.read_excel("data/Training_dataset.xlsx")
df_test = pd.read_excel("data/Testing_dataset.xlsx")

print("=" * 60)
print("FILE TRAINING")
print("=" * 60)
print("Kích thước:", df_train.shape, "(dòng, cột)")
print("Tên các cột:", df_train.columns.tolist())
print("\n5 dòng đầu tiên:")
print(df_train.head())
print("\nThông tin kiểu dữ liệu:")
print(df_train.dtypes)
print("\nSố giá trị thiếu mỗi cột:")
print(df_train.isnull().sum())

print("\n" + "=" * 60)
print("FILE TESTING")
print("=" * 60)
print("Kích thước:", df_test.shape, "(dòng, cột)")
print("Tên các cột:", df_test.columns.tolist())
print("\n5 dòng đầu tiên:")
print(df_test.head())

# Nếu có cột label, in phân bố nhãn
for col in df_train.columns:
    if df_train[col].nunique() <= 5 and df_train[col].dtype in ['int64', 'float64']:
        print(f"\nPhân bố nhãn trong cột '{col}':")
        print(df_train[col].value_counts())