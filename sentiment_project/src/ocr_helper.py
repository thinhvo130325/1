"""
Helper: OCR ảnh review + xuất kết quả thành ảnh PNG.
"""
import os
import re
from PIL import Image, ImageDraw, ImageFont
import pytesseract

# ============================================================
# CẤU HÌNH TESSERACT (Windows)
# ============================================================
# Sửa đường dẫn nếu bạn cài ở chỗ khác
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


def ocr_image(image: Image.Image, lang: str = "vie") -> str:
    """
    Đọc chữ từ ảnh.
    - lang="vie": tiếng Việt
    - lang="vie+eng": tiếng Việt + Anh (nếu có teencode tiếng Anh)
    """
    try:
        text = pytesseract.image_to_string(image, lang=lang)
    except pytesseract.TesseractNotFoundError:
        raise RuntimeError(
            "Không tìm thấy Tesseract. Cài tại "
            "https://github.com/UB-Mannheim/tesseract/wiki "
            "và kiểm tra đường dẫn TESSERACT_PATH."
        )
    return text


def split_into_comments(text: str) -> list[str]:
    """
    Tách văn bản OCR thành từng bình luận riêng lẻ.
    Chiến lược đơn giản: tách theo dòng, lọc dòng quá ngắn.
    """
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    # Giữ dòng có ít nhất 3 ký tự chữ (loại dòng chỉ có số, ký tự lạ)
    valid = [l for l in lines if len(re.findall(r"\w", l)) >= 3]
    return valid


def draw_results_on_image(
    image: Image.Image,
    results: list[dict],
    output_path: str = "ket_qua.png"
) -> str:
    """
    Vẽ kết quả dự đoán lên ảnh gốc.
    results: [{"text": "...", "label": "TÍCH CỰC", "confidence": 0.9}, ...]

    Vẽ khung màu xanh (tích cực) / đỏ (tiêu cực) + nhãn ở góc ảnh.
    """
    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img)

    # Font — fallback nếu không có arial
    try:
        font = ImageFont.truetype("arial.ttf", size=20)
        font_small = ImageFont.truetype("arial.ttf", size=16)
    except OSError:
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # Tổng kết
    n_pos = sum(1 for r in results if "TÍCH CỰC" in r["label"])
    n_neg = len(results) - n_pos
    summary = f"Tổng: {len(results)} câu  |  Tích cực: {n_pos}  |  Tiêu cực: {n_neg}"

    # Vẽ header
    draw.rectangle([(0, 0), (img.width, 60)], fill=(30, 30, 30))
    draw.text((15, 18), "KẾT QUẢ PHÂN TÍCH CẢM XÚC", fill="white", font=font)
    draw.text((15, 42), summary, fill=(200, 200, 200), font=font_small)

    # Vẽ từng câu kết quả ở footer
    footer_h = 30 * len(results) + 20
    new_img = Image.new("RGB", (img.width, img.height + footer_h), (255, 255, 255))
    new_img.paste(img, (0, 0))

    draw = ImageDraw.Draw(new_img)
    y = img.height + 10
    for r in results:
        is_pos = "TÍCH CỰC" in r["label"]
        color = (0, 128, 0) if is_pos else (200, 0, 0)
        icon = "✓" if is_pos else "✗"
        conf = f"{r['confidence']:.0%}" if r.get("confidence") else "N/A"
        text = r["text"][:70] + ("..." if len(r["text"]) > 70 else "")
        draw.text((10, y), f"{icon} [{conf}] {text}", fill=color, font=font_small)
        y += 28

    new_img.save(output_path)
    return output_path