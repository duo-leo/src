import re
from underthesea import text_normalize

from utils.constant import VIETNAMESE_VOWEL

punctuation_list = ['.', ',', '?', '!', ':', ';', '(', ')', '[', ']', '{', '}', '<', '>', '"', "'", '“', '”', '‘', '’', '…', '—',
                    '。', '，', '？', '！', '：', '；', '（', '）', '【', '】', '「', '」', '《', '》', '〈', '〉', '“', '”', '‘', '’', '…', '—', '–']

def normalize_chinese_line(chinese) -> str:
    chinese = re.sub(r'[^\w]', '', chinese)
    # remove any digit in chinese
    chinese = re.sub(r'\d', '', chinese)
    return chinese

def normalize_vietnamese_line(vietnamese) -> str:
    vietnamese = text_normalize(vietnamese)
    vietnamese = vietnamese.lower().strip().replace('  ', ' ')
    vietnamese = re.sub(r'[^\w\s]', ' ', vietnamese)
    vietnamese = vietnamese.replace('  ', ' ').lower().strip()
    vietnamese = vietnamese.replace('  ', ' ').strip()
    return vietnamese

def normalize_vietnamese(text):
    telex_table = str.maketrans({
        'ă': 'aw', 'â': 'aa', 'ê': 'ee', 'ô': 'oo', 'ơ': 'ow', 'ư': 'uw',
        'Ă': 'AW', 'Â': 'AA', 'Ê': 'EE', 'Ô': 'OO', 'Ơ': 'OW', 'Ư': 'UW',
    })

    vni_table = str.maketrans({
        'á': 'a1', 'à': 'a2', 'ả': 'a3', 'ã': 'a4', 'ạ': 'a5',
        'ắ': 'aw1', 'ằ': 'aw2', 'ẳ': 'aw3', 'ẵ': 'aw4', 'ặ': 'aw5',
        'ấ': 'aa1', 'ầ': 'aa2', 'ẩ': 'aa3', 'ẫ': 'aa4', 'ậ': 'aa5',
        'é': 'e1', 'è': 'e2', 'ẻ': 'e3', 'ẽ': 'e4', 'ẹ': 'e5',
        'ế': 'ee1', 'ề': 'ee2', 'ể': 'ee3', 'ễ': 'ee4', 'ệ': 'ee5',
        'í': 'i1', 'ì': 'i2', 'ỉ': 'i3', 'ĩ': 'i4', 'ị': 'i5',
        'ó': 'o1', 'ò': 'o2', 'ỏ': 'o3', 'õ': 'o4', 'ọ': 'o5',
        'ố': 'oo1', 'ồ': 'oo2', 'ổ': 'oo3', 'ỗ': 'oo4', 'ộ': 'oo5',
        'ớ': 'ow1', 'ờ': 'ow2', 'ở': 'ow3', 'ỡ': 'ow4', 'ợ': 'ow5',
        'ú': 'u1', 'ù': 'u2', 'ủ': 'u3', 'ũ': 'u4', 'ụ': 'u5',
        'ứ': 'uw1', 'ừ': 'uw2', 'ử': 'uw3', 'ữ': 'uw4', 'ự': 'uw5',
        'ý': 'y1', 'ỳ': 'y2', 'ỷ': 'y3', 'ỹ': 'y4', 'ỵ': 'y5',
        'đ': 'dd',
        'Á': 'A1', 'À': 'A2', 'Ả': 'A3', 'Ã': 'A4', 'Ạ': 'A5',
        'Ắ': 'AW1', 'Ằ': 'AW2', 'Ẳ': 'AW3', 'Ẵ': 'AW4', 'Ặ': 'AW5',
        'Ấ': 'AA1', 'Ầ': 'AA2', 'Ẩ': 'AA3', 'Ẫ': 'AA4', 'Ậ': 'AA5',
        'É': 'E1', 'È': 'E2', 'Ẻ': 'E3', 'Ẽ': 'E4', 'Ẹ': 'E5',
        'Ế': 'EE1', 'Ề': 'EE2', 'Ể': 'EE3', 'Ễ': 'EE4', 'Ệ': 'EE5',
        'Í': 'I1', 'Ì': 'I2', 'Ỉ': 'I3', 'Ĩ': 'I4', 'Ị': 'I5',
        'Ó': 'O1', 'Ò': 'O2', 'Ỏ': 'O3', 'Õ': 'O4', 'Ọ': 'O5',
        'Ố': 'OO1', 'Ồ': 'OO2', 'Ổ': 'OO3', 'Ỗ': 'OO4', 'Ộ': 'OO5',
        'Ớ': 'OW1', 'Ờ': 'OW2', 'Ở': 'OW3', 'Ỡ': 'OW4', 'Ợ': 'OW5',
        'Ú': 'U1', 'Ù': 'U2', 'Ủ': 'U3', 'Ũ': 'U4', 'Ụ': 'U5',
        'Ứ': 'UW1', 'Ừ': 'UW2', 'Ử': 'UW3', 'Ữ': 'UW4', 'Ự': 'UW5',
        'Ý': 'Y1', 'Ỳ': 'Y2', 'Ỷ': 'Y3', 'Ỹ': 'Y4', 'Ỵ': 'Y5',
        'Đ': 'DD'
    })
    
    # Normalize remaining Unicode characters using Telex
    normalized_text = text.translate(telex_table)

    # Normalize Unicode characters using VNI
    normalized_text = normalized_text.translate(vni_table)

    # Move all numbers to the end of each word
    tmp = ""
    for a in normalized_text.split():
        tmp += ''.join([i for i in a if not i.isdigit()]) + ''.join([i for i in a if i.isdigit()]) + " "
    normalized_text = tmp.strip()

    # Replace spaces with underscores
    normalized_text = normalized_text.replace(' ', '_')
    
    return normalized_text

def denormalize_vietnamese(text):
    telex_table = {
        'aw': 'ă', 'aa': 'â', 'ee': 'ê', 'oo': 'ô', 'ow': 'ơ', 'uw': 'ư', 'dd': 'đ',
        'AW': 'Ă', 'AA': 'Â', 'EE': 'Ê', 'OO': 'Ô', 'OW': 'Ơ', 'UW': 'Ư', 'DD': 'Đ'
    }

    vni_table = {
        'a1': 'á', 'a2': 'à', 'a3': 'ả', 'a4': 'ã', 'a5': 'ạ', 'a0': 'a',
        'ă1': 'ắ', 'ă2': 'ằ', 'ă3': 'ẳ', 'ă4': 'ẵ', 'ă5': 'ặ', 'ă0': 'ă',
        'â1': 'ấ', 'â2': 'ầ', 'â3': 'ẩ', 'â4': 'ẫ', 'â5': 'ậ', 'â0': 'â',
        'e1': 'é', 'e2': 'è', 'e3': 'ẻ', 'e4': 'ẽ', 'e5': 'ẹ', 'e0': 'e',
        'ê1': 'ế', 'ê2': 'ề', 'ê3': 'ể', 'ê4': 'ễ', 'ê5': 'ệ', 'ê0': 'ê',
        'i1': 'í', 'i2': 'ì', 'i3': 'ỉ', 'i4': 'ĩ', 'i5': 'ị', 'i0': 'i',
        'o1': 'ó', 'o2': 'ò', 'o3': 'ỏ', 'o4': 'õ', 'o5': 'ọ', 'o0': 'o',
        'ô1': 'ố', 'ô2': 'ồ', 'ô3': 'ổ', 'ô4': 'ỗ', 'ô5': 'ộ', 'ô0': 'ô',
        'ơ1': 'ớ', 'ơ2': 'ờ', 'ơ3': 'ở', 'ơ4': 'ỡ', 'ơ5': 'ợ', 'ơ0': 'ơ',
        'u1': 'ú', 'u2': 'ù', 'u3': 'ủ', 'u4': 'ũ', 'u5': 'ụ', 'u0': 'u',
        'ư1': 'ứ', 'ư2': 'ừ', 'ư3': 'ử', 'ư4': 'ữ', 'ư5': 'ự', 'ư0': 'ư',
        'y1': 'ý', 'y2': 'ỳ', 'y3': 'ỷ', 'y4': 'ỹ', 'y5': 'ỵ', 'y0': 'y'
    }

    for k, v in telex_table.items():
        text = text.replace(k, v)

    tmp = text.split('_')
    # print(tmp)
    for i in range(len(tmp)):
        if tmp[i][-1].isdigit():
            for j in range(len(tmp[i]) - 1, 0, -1):
                if tmp[i][j] in VIETNAMESE_VOWEL:
                    tmp[i] = tmp[i][:j + 1] + tmp[i][-1] + tmp[i][j + 1:-1]
                    break

    text = ' '.join(tmp)

    for k, v in vni_table.items():
        text = text.replace(k, v)

    text = text_normalize(text)
    return text
    