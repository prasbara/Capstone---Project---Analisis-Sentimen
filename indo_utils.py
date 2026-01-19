import re
import pandas as pd
import emoji

# Kamus kata tidak baku (sederhana, bisa diperluas)
NORMALIZATION_DICT = {
    "gk": "tidak", "gak": "tidak", "ga": "tidak", "nggak": "tidak",
    "bgt": "banget", "yg": "yang", "kalo": "kalau", "kl": "kalau",
    "dgn": "dengan", "tdk": "tidak", "jgn": "jangan", "krn": "karena",
    "krna": "karena", "sdh": "sudah", "udh": "sudah", "blm": "belum",
    "tau": "tahu", "klo": "kalau", "tp": "tapi", "ak": "aku",
    "bs": "bisa", "jd": "jadi", "jdi": "jadi", "bkn": "bukan",
    "dr": "dari", "utk": "untuk", "skrg": "sekarang", "trs": "terus",
    "tpi": "tapi", "kpn": "kapan", "dpt": "dapat", "br": "baru",
    "pke": "pakai", "pake": "pakai", "km": "kamu", "sy": "saya",
    "syg": "sayang", "bwh": "bawah", "atas": "atas", "dlm": "dalam",
    "bpk": "bapak", "ibu": "ibu", "kak": "kakak", "min": "admin",
    "mhn": "mohon", "maaf": "maaf", "makasih": "terima kasih",
    "tq": "terima kasih", "thx": "terima kasih", "d": "di",
    "emg": "memang", "emang": "memang", "aj": "saja", "aja": "saja",
    "ni": "ini", "tu": "itu", "sm": "sama", "smp": "sampai"
}

def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # 3. Remove Mentions (@username)
    text = re.sub(r'@\w+', '', text)
    
    # 4. Remove Hashtags
    text = re.sub(r'#\w+', '', text)
    
    # 5. Remove Emoji
    text = emoji.replace_emoji(text, replace='')
    
    # 6. Remove Numbers (optional, but good for sentiment if context allows)
    text = re.sub(r'\d+', '', text)
    
    # 7. Remove Punctuation & Special Characters
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # 8. Remove Multiple Whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def normalize_text(text):
    if not isinstance(text, str):
        return ""
    
    words = text.split()
    normalized_words = [NORMALIZATION_DICT.get(w, w) for w in words]
    return " ".join(normalized_words)

def preprocess_pipeline(text):
    text = clean_text(text)
    text = normalize_text(text)
    return text

# Valid Positive/Negative Words (Indonesian + Javanese)
POSITIVE_KEYWORDS = {
    # General
    'keren', 'mantap', 'terima kasih', 'semangat', 'maju', 'alhamdulillah', 'bagus', 'nyata', 'kerja nyata', 'setuju', 'gercep',
    'cantik', 'indah', 'sukses', 'hebat', 'seru', 'diperhatikan', 'disiapkan', 'waspada', 'dihimbau', 'berfungsi', 'siap',
    'aman', 'sigap', 'cepat', 'tanggap', 'terpantau', 'terkendali', 'lancar', 'responsif', 'peduli',
    
    # Javanese / Slang
    'jos', 'josjis', 'apik', 'sae', 'maturnuwun', 'suwun', 'gas', 'gaspol', 'gayeng', 'beja', 'penak', 'istimewa', 'menyala',
    'manut', 'sat set', 'ayu', 'mberkahi', 'top', 'petcah', 'digatekke', 'wis disiapke', 'waspodo', 'diimbau', 'iso mlaku',
    'siap', 'aman', 'cekatan', 'cepet', 'tanggap', 'ketok', 'terkendali', 'lancar', 'responsif', 'peduli'
}

NEGATIVE_KEYWORDS = {
    # General
    'banjir', 'macet', 'rusak', 'hancur', 'lambat', 'lelet', 'bohong', 'hoax', 'pencitraan', 'bodoh', 'tidak berguna', 'malu',
    'bau', 'asap hitam', 'diam', 'aneh', 'kotor', 'parah', 'sia-sia', 'percuma', 'omong kosong', 'kasar', 'panas', 'kenapa',
    'selalu', 'bermasalah', 'tidak berfungsi', 'terlambat', 'gagal', 'diabaikan', 'tidak jelas', 'kurang',
    
    # Javanese / Slang
    'kelelep', 'asat', 'mendet', 'bubrah', 'remuk', 'ajur', 'suwe', 'kesuwen', 'ngapusi', 'dobol', 'gedabrus', 'cangkeman',
    'atur-atur', 'pekok', 'goblok', 'ra mutu', 'isin', 'banger', 'cumi-cumi', 'cumi darat', 'meneng tok', 'turu', 'wagu',
    'ra ceto', 'kemproh', 'kumuh', 'ngenes', 'muspro', 'ra guna', 'pret', 'mbelgedes', 'gathel', 'telek', 'bajilak', 'sumuk',
    'ngentang-ngentang', 'ngopo', 'terus wae', 'masalah', 'ora fungsi', 'telat', 'alon', 'muspra', 'ora digatekke', 'ora cetho'
}

NEUTRAL_KEYWORDS = {
    # General
    'info', 'informasi', 'tanya', 'bertanya', 'kapan', 'dimana', 'siapa', 'bagaimana', 'semoga', 'tolong', 'mohon', 'saran',
    'jam berapa', 'lewat', 'melintas', 'turut berduka cita', 'innalillahi', 'cuaca', 'hujan', 'lapor', 'hadir', 'absen',
    'pompa', 'saluran air', 'cek', 'amati', 'disiapkan', 'diperhatikan', 'masyarakat', 'dihimbau',
    
    # Javanese / Slang
    'nyuwun sewu', 'takon', 'mben', 'nandi', 'ngendi', 'sopo', 'piye', 'pripun', 'mugi', 'mugi-mugi', 'tulung', 'usul',
    'jam piro', 'liwat', 'nderek belasungkawa', 'husnul khotimah', 'udan', 'terang', 'wadul', 'nyimak', 'saluran banyu',
    'dicek', 'diamati', 'disiapke', 'digatekke', 'diimbau', 'pengen', 'mau', 'minat', 'coba', 'njajal'
}
