import re
import unicodedata
from pyvi import ViTokenizer

# ============================================================
# TỪ ĐIỂN TEENCODE / VIẾT TẮT PHỔ BIẾN (tiếng Việt)
# ============================================================
TEENCODE_MAP = {
    # Phủ định
    "ko": "không", "k": "không", "kh": "không", "kg": "không",
    "hk": "không", "hok": "không", "hông": "không", "khum": "không",
    "hem": "không", "chả": "không", "chẳng": "không",
    # Động từ / trạng từ phổ biến
    "đc": "được", "dc": "được", "dk": "được", "đk": "được",
    "đt": "điện thoại", "nv": "nhân viên", "sp": "sản phẩm",
    "vs": "với", "j": "gì", "r": "rồi", "ns": "nói",
    "bt": "bình thường", "cx": "cũng", "cg": "cũng",
    "trc": "trước", "mn": "mọi người", "ae": "anh em",
    "tk": "tài khoản", "ib": "nhắn tin", "rep": "trả lời",
    "sll": "số lượng lớn", "sl": "số lượng", "vc": "vận chuyển",
    "gd": "giao dịch", "tks": "cảm ơn", "thanks": "cảm ơn",
    "ok": "ổn", "oke": "ổn", "okie": "ổn", "okela": "ổn",
    "sđt": "số điện thoại", "ship": "giao hàng", "shipper": "người giao hàng",
    "lm": "làm", "bik": "biết", "bít": "biết", "hĩu": "hiểu",
    "cute": "dễ thương", "xink": "xinh", "xinhh": "xinh",
    "đx": "được", "thik": "thích", "thjk": "thích",
    "sản xuất": "sản xuất", "qá": "quá", "wá": "quá",
    "wa": "quá", "qa": "quá", "nhìu": "nhiều", "nhju": "nhiều",
    "z": "vậy", "zậy": "vậy", "dzậy": "vậy",
    "iu": "yêu", "yeu": "yêu",
}

# ============================================================
# EMOJI → CẢM XÚC (thay vì xóa, chuyển thành từ)
# ============================================================
EMOJI_POSITIVE = {
    "😍", "😘", "❤️", "❤", "💕", "💖", "👍", "👌", "😊", "😁",
    "🥰", "😻", "💯", "✨", "🎉", "😄", "😃", "🙂", "☺️", "😀",
    "⭐⭐⭐⭐⭐", "⭐", "🌟", "🔥", "👏", "🙌", "💪",
}
EMOJI_NEGATIVE = {
    "😡", "😠", "👎", "💔", "😞", "😢", "😭", "😤", "🤬", "🙄",
    "😑", "😒", "😖", "😣", "😫", "😩", "🤮", "🤢", "💩", "⚠️",
}

STOPWORDS = set([
    "thì", "là", "của", "và", "có", "cho", "một", "các", "những", "được",
    "trong", "đã", "đang", "sẽ", "bị", "bởi", "với", "để", "này", "đó",
    "khi", "như", "từ", "ra", "vào", "lên", "xuống", "rồi", "nữa",
    "ạ", "à", "ơi", "nhé", "nha", "vậy", "thế", "mà", "nên", "vì",
])


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def normalize_elongation(text: str) -> str:
    """Giảm ký tự lặp: 'tốtttttt' → 'tốt', 'đẹppppp' → 'đẹp', giữ 'hihi'."""
    # Ký tự lặp >= 3 lần → giữ 1
    return re.sub(r"(.)\1{2,}", r"\1", text)


def handle_emoji(text: str) -> str:
    """Chuyển emoji thành từ cảm xúc (thay vì xóa)."""
    for emo in EMOJI_POSITIVE:
        text = text.replace(emo, " emoji_tich_cuc ")
    for emo in EMOJI_NEGATIVE:
        text = text.replace(emo, " emoji_tieu_cuc ")
    # Xóa các emoji còn lại (neutral)
    text = re.sub(r"[\U0001F300-\U0001FAFF\U00002600-\U000027BF]+", " ", text)
    return text


def expand_teencode(text: str) -> str:
    """Thay teencode/viết tắt bằng từ chuẩn. Xử lý theo từ (word boundary)."""
    tokens = text.split()
    out = []
    for tok in tokens:
        clean = tok.strip(".,!?;:()[]{}'\"")
        replacement = TEENCODE_MAP.get(clean.lower())
        if replacement:
            out.append(replacement)
        else:
            out.append(tok)
    return " ".join(out)


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_vietnamese(text: str) -> str:
    return ViTokenizer.tokenize(text)


def remove_stopwords(text: str) -> str:
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS]
    return " ".join(tokens)


def preprocess_pipeline(text: str) -> str:
    """
    Pipeline đầy đủ, đã nâng cấp để chịu được data thực tế:
      1. Chuẩn hóa Unicode
      2. Xử lý emoji → từ cảm xúc
      3. Lowercase + bỏ ký tự đặc biệt
      4. Mở rộng teencode/viết tắt
      5. Giảm ký tự lặp (elongation)
      6. Tách từ tiếng Việt
      7. Bỏ stopwords
    """
    text = normalize_unicode(text)
    text = handle_emoji(text)          # 🔥 MỚI
    text = clean_text(text)
    text = expand_teencode(text)       # 🔥 MỚI
    text = normalize_elongation(text)  # 🔥 MỚI
    text = tokenize_vietnamese(text)
    text = remove_stopwords(text)
    return text