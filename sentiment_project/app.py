"""
Streamlit app: Phân tích cảm xúc bình luận tiếng Việt (tối ưu tốc độ).
Chạy: python -m streamlit run app.py
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from demo import SentimentPredictor
from preprocess import preprocess_pipeline
from ocr_helper import ocr_image, split_into_comments, draw_results_on_image


# ============================================================
# CẤU HÌNH
# ============================================================
st.set_page_config(
    page_title="Sentiment Analysis - Tiếng Việt",
    page_icon="💬",
    layout="wide",
)

PAGE_SIZE = 50          # số dòng/trang
BATCH_SIZE = 500         # số câu predict 1 lần (để có progress bar)


# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_predictor():
    return SentimentPredictor("linear_svc")


try:
    predictor = load_predictor()
except FileNotFoundError:
    st.error("⚠️ Không tìm thấy model. Hãy chạy `python src/train.py` trước.")
    st.stop()


# ============================================================
# PREDICT BATCH (nhanh gấp 20–50 lần so với predict từng câu)
# ============================================================
def predict_batch(texts: list) -> list:
    """
    Predict 1 lần cho cả list câu.
    Trả về list dict gọn: {Câu, Dự đoán, Độ tin cậy, pred}
    (Không lưu _raw để session_state nhẹ + render nhanh.)
    """
    if not texts:
        return []

    # Preprocess từng câu (vẫn tuần tự vì pyvi, nhưng không có overhead Streamlit)
    processed = [preprocess_pipeline(str(t)) for t in texts]

    # Vectorize + predict 1 lần → nhanh hơn rất nhiều
    X = predictor.vectorizer.transform(processed)
    preds = predictor.model.predict(X)

    # Độ tin cậy
    if hasattr(predictor.model, "predict_proba"):
        confs = np.max(predictor.model.predict_proba(X), axis=1)
    elif hasattr(predictor.model, "decision_function"):
        scores = predictor.model.decision_function(X)
        confs = 1.0 / (1.0 + np.exp(-np.abs(scores)))
    else:
        confs = np.ones(len(texts))

    return [
        {
            "Câu": str(t),
            "Dự đoán": "TÍCH CỰC 😊" if int(p) == 1 else "TIÊU CỰC 😞",
            "Độ tin cậy": f"{float(c):.1%}",
            "pred": int(p),
        }
        for t, p, c in zip(texts, preds, confs)
    ]


def predict_batch_with_progress(texts: list, progress_bar) -> list:
    """Predict theo lô 500 câu để có progress bar."""
    results = []
    total = len(texts)
    t0 = time.time()
    for i in range(0, total, BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        results.extend(predict_batch(batch))
        done = min(i + BATCH_SIZE, total)
        progress_bar.progress(
            done / total,
            text=f"Đang xử lý {done}/{total}... ({time.time() - t0:.1f}s)",
        )
    return results


# ============================================================
# TIỆN ÍCH HIỂN THỊ
# ============================================================
def dem_ket_qua(results: list) -> tuple:
    n_pos = sum(1 for r in results if r["pred"] == 1)
    return n_pos, len(results) - n_pos


def loc_theo_nhan(results: list, lua_chon: str) -> list:
    if lua_chon == "Chỉ tích cực ✅":
        return [r for r in results if r["pred"] == 1]
    if lua_chon == "Chỉ tiêu cực ❌":
        return [r for r in results if r["pred"] == 0]
    return results


def hien_thi_ket_qua_theo_nhan(results: list, key_prefix: str):
    """
    Hiển thị tổng hợp + bộ lọc + pagination + tải CSV.
    Đã tối ưu: chỉ render 50 dòng/lần, bullet chỉ khi tick checkbox.
    """
    if not results:
        st.warning("Không có kết quả để hiển thị.")
        return

    # --- 1. Metric tổng quan ---
    n_pos, n_neg = dem_ket_qua(results)
    c1, c2, c3 = st.columns(3)
    c1.metric("📊 Tổng số câu", len(results))
    c2.metric("✅ Tích cực", n_pos)
    c3.metric("❌ Tiêu cực", n_neg)
    st.divider()

    # --- 2. Bộ lọc ---
    st.subheader("🔎 Lọc và xem chi tiết")
    lua_chon = st.radio(
        "Chọn nhãn muốn xem:",
        ["Tất cả", "Chỉ tích cực ✅", "Chỉ tiêu cực ❌"],
        horizontal=True,
        key=f"filter_{key_prefix}",
    )

    filtered = loc_theo_nhan(results, lua_chon)
    st.caption(f"Đang hiển thị **{len(filtered)}** / {len(results)} câu.")

    if not filtered:
        st.info(f"Không có câu nào thuộc nhóm **{lua_chon}**.")
        return

    # --- 3. Pagination ---
    total_pages = max(1, (len(filtered) + PAGE_SIZE - 1) // PAGE_SIZE)
    if total_pages > 1:
        col_p1, col_p2 = st.columns([1, 3])
        with col_p1:
            page = st.number_input(
                "Trang:",
                min_value=1,
                max_value=total_pages,
                value=1,
                step=1,
                key=f"page_{key_prefix}_{lua_chon}",
            )
        with col_p2:
            st.caption(
                f"Trang {page}/{total_pages} — {PAGE_SIZE} câu/trang — "
                f"tổng {len(filtered)} câu"
            )
    else:
        page = 1

    start = (page - 1) * PAGE_SIZE
    end = min(start + PAGE_SIZE, len(filtered))
    page_results = filtered[start:end]

    # --- 4. Bảng (chỉ trang hiện tại) ---
    df_page = pd.DataFrame([
        {"Câu": r["Câu"], "Dự đoán": r["Dự đoán"], "Độ tin cậy": r["Độ tin cậy"]}
        for r in page_results
    ])
    st.dataframe(df_page, use_container_width=True, hide_index=True, height=380)

    # --- 5. Bullet chỉ khi tick (tránh render 11k dòng) ---
    if st.checkbox(
        f"📋 Xem dạng danh sách ({len(page_results)} câu ở trang này)",
        key=f"show_bullets_{key_prefix}_{lua_chon}_{page}",
    ):
        for i, r in enumerate(page_results, start=start + 1):
            icon = "✅" if r["pred"] == 1 else "❌"
            st.markdown(f"**{i}.** {icon} `[{r['Độ tin cậy']}]` {r['Câu']}")

    # --- 6. Tải CSV (toàn bộ filtered, không chỉ trang) ---
    df_full = pd.DataFrame([
        {"Câu": r["Câu"], "Dự đoán": r["Dự đoán"], "Độ tin cậy": r["Độ tin cậy"]}
        for r in filtered
    ])
    csv = df_full.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    fname = (
        "ket_qua_tich_cuc.csv" if lua_chon == "Chỉ tích cực ✅"
        else "ket_qua_tieu_cuc.csv" if lua_chon == "Chỉ tiêu cực ❌"
        else "ket_qua_tat_ca.csv"
    )
    st.download_button(
        f"⬇️ Tải CSV ({len(filtered)} câu — {lua_chon.lower()})",
        csv,
        file_name=fname,
        mime="text/csv",
        key=f"dl_{key_prefix}_{lua_chon}",
    )


def nut_tai_anh(img: Image.Image, results: list, filename: str, key: str):
    """Vẽ kết quả lên ảnh + nút tải PNG."""
    # Chỉ vẽ tối đa 30 câu đầu để ảnh không quá dài
    out_path = f"_{filename}.png"
    draw_results_on_image(img, results, out_path)
    with open(out_path, "rb") as f:
        st.download_button(
            "🖼️ Tải ảnh kết quả (PNG)",
            f.read(),
            file_name=filename,
            mime="image/png",
            key=f"dl_png_{key}",
        )


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("ℹ️ Thông tin")
    st.write("**Mô hình:** Linear SVC")
    st.write("**Accuracy:** 95.53%")
    st.write("**Đặc trưng:** TF-IDF (12.212 chiều)")
    st.write("**Huấn luyện:** 11.409 mẫu")
    st.divider()
    st.caption(f"⚡ Batch size: {BATCH_SIZE} | Trang: {PAGE_SIZE} dòng")


# ============================================================
# HEADER
# ============================================================
st.title("💬 Phân tích Cảm xúc Bình luận Tiếng Việt")
st.caption(
    "Nhập câu, dán ảnh review, hoặc chụp màn hình — AI sẽ phân tích Tích cực / Tiêu cực."
)


# ============================================================
# 4 TABS
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📝 Nhập câu",
    "🖼️ Upload ảnh (OCR)",
    "📸 Chụp màn hình",
    "📊 Batch CSV",
])


# -------------------- TAB 1 --------------------
with tab1:
    st.subheader("Nhập một câu bình luận")
    text = st.text_area(
        "Nội dung:",
        height=120,
        placeholder="VD: Sản phẩm rất tốt, giao hàng nhanh, đóng gói cẩn thận",
        key="input_text_tab1",
    )

    if st.button("🔍 Phân tích", type="primary", key="btn_tab1"):
        if not text.strip():
            st.warning("Vui lòng nhập câu.")
        else:
            r = predictor.predict(text)
            is_pos = r["pred"] == 1

            col1, col2 = st.columns(2)
            col1.metric("Kết quả", r["label"])
            conf = r.get("confidence")
            col2.metric("Độ tin cậy", f"{conf:.1%}" if conf else "N/A")

            st.success(f"✅ {r['label']}") if is_pos else st.error(f"❌ {r['label']}")

            with st.expander("Xem câu sau tiền xử lý"):
                st.code(r["processed"], language="text")


# -------------------- TAB 2: UPLOAD ẢNH --------------------
with tab2:
    st.subheader("Upload ảnh chụp review")
    uploaded = st.file_uploader(
        "Chọn ảnh:", type=["png", "jpg", "jpeg", "webp"], key="upload_img"
    )

    if uploaded:
        img = Image.open(uploaded)
        col_img, col_res = st.columns([1, 1])

        with col_img:
            st.image(img, caption="Ảnh gốc", use_container_width=True)

        with col_res:
            try:
                with st.spinner("Đang OCR..."):
                    ocr_text = ocr_image(img, lang="vie")
            except Exception as e:
                st.error(f"Lỗi OCR: {e}")
                ocr_text = ""

            edited_text = st.text_area(
                "Văn bản OCR (sửa nếu sai):",
                value=ocr_text,
                height=200,
                key="edited_ocr_tab2",
            )

            if st.button("📊 Phân tích", type="primary", key="btn_ocr_tab2"):
                comments = split_into_comments(edited_text)
                if not comments:
                    st.warning("Không tìm thấy bình luận nào.")
                else:
                    with st.spinner(f"Đang phân tích {len(comments)} câu..."):
                        results = predict_batch(comments)
                    st.session_state["results_tab2"] = results
                    st.session_state["img_tab2"] = img

    if "results_tab2" in st.session_state:
        st.divider()
        hien_thi_ket_qua_theo_nhan(st.session_state["results_tab2"], "tab2")
        st.divider()
        nut_tai_anh(
            st.session_state["img_tab2"],
            st.session_state["results_tab2"],
            filename="ket_qua_phan_tich.png",
            key="tab2",
        )


# -------------------- TAB 3: CHỤP MÀN HÌNH --------------------
with tab3:
    st.subheader("📸 Chụp màn hình review")

    mode = st.radio(
        "Cách chụp:",
        ["Webcam", "Chụp màn hình máy tính (upload)"],
        horizontal=True,
        key="mode_tab3",
    )

    img_capture = None
    if mode == "Webcam":
        img_capture = st.camera_input("Chụp từ webcam:", key="cam_tab3")
    else:
        st.info(
            "💡 **Windows:** `Win + Shift + S` → `Ctrl + V` vào Paint → lưu file → upload.\n\n"
            "**macOS:** `Cmd + Shift + 4` → file tự lưu ở Desktop."
        )
        img_capture = st.file_uploader(
            "Upload ảnh chụp màn hình:",
            type=["png", "jpg", "jpeg"],
            key="screenshot_tab3",
        )

    if img_capture:
        img = Image.open(img_capture)
        col_img, col_res = st.columns([1, 1])

        with col_img:
            st.image(img, caption="Ảnh đã chụp", use_container_width=True)

        with col_res:
            try:
                with st.spinner("Đang OCR..."):
                    ocr_text = ocr_image(img, lang="vie")
            except Exception as e:
                st.error(f"Lỗi OCR: {e}")
                ocr_text = ""

            edited_text = st.text_area(
                "Văn bản OCR (sửa nếu sai):",
                value=ocr_text,
                height=250,
                key="edit_screenshot_tab3",
            )

            if st.button("📊 Phân tích", type="primary", key="btn_screenshot_tab3"):
                comments = split_into_comments(edited_text)
                if not comments:
                    st.warning("Không tìm thấy bình luận nào.")
                else:
                    with st.spinner(f"Đang phân tích {len(comments)} câu..."):
                        results = predict_batch(comments)
                    st.session_state["results_tab3"] = results
                    st.session_state["img_tab3"] = img

    if "results_tab3" in st.session_state:
        st.divider()
        hien_thi_ket_qua_theo_nhan(st.session_state["results_tab3"], "tab3")
        st.divider()
        nut_tai_anh(
            st.session_state["img_tab3"],
            st.session_state["results_tab3"],
            filename="ket_qua_chup_man_hinh.png",
            key="tab3",
        )


# -------------------- TAB 4: BATCH CSV --------------------
with tab4:
    st.subheader("Phân tích hàng loạt từ file CSV")
    st.write("File CSV cần **có ít nhất 1 cột chứa bình luận**.")

    uploaded_csv = st.file_uploader("Chọn file CSV:", type=["csv"], key="csv_batch")

    if uploaded_csv:
        df_in = pd.read_csv(uploaded_csv)
        st.write(f"**Đã đọc {len(df_in)} dòng.** Cột: `{df_in.columns.tolist()}`")

        text_col = st.selectbox(
            "Chọn cột chứa bình luận:",
            df_in.columns.tolist(),
            key="col_batch",
        )

        # Nút "Xóa kết quả cũ" để giải phóng session_state nếu cần
        if "results_tab4" in st.session_state:
            if st.button("🗑️ Xóa kết quả cũ", key="clear_tab4"):
                del st.session_state["results_tab4"]
                st.rerun()

        if st.button("🚀 Phân tích tất cả", type="primary", key="btn_batch"):
            texts = df_in[text_col].astype(str).tolist()
            prog = st.progress(0.0, text="Bắt đầu...")
            t0 = time.time()
            results = predict_batch_with_progress(texts, prog)
            elapsed = time.time() - t0
            st.session_state["results_tab4"] = results
            st.success(f"✅ Xong {len(results)} câu trong **{elapsed:.1f}s** "
                       f"(~{len(results)/elapsed:.0f} câu/giây)")

    if "results_tab4" in st.session_state:
        st.divider()
        hien_thi_ket_qua_theo_nhan(st.session_state["results_tab4"], "tab4")


# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption("Đồ án Phân tích Cảm xúc — TF-IDF + Linear SVC — Demo Streamlit")