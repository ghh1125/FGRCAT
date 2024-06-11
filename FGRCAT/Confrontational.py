import nltk
from nltk.corpus import wordnet

# 设置 NLTK 数据路径
nltk.data.path.append('/home/amax/nltk_data')

# 查看 NLTK 数据路径，确认已经包含了我们上传的路径
print(nltk.data.path)

# 确保 WordNet 资源已经加载
try:
    wordnet.ensure_loaded()
except LookupError:
    print("WordNet resource not found. Please make sure it is placed in the correct directory.")

# 导入其他必要的模块
from nltk.corpus import wordnet

# 定义一个函数来获取词语的同义词
def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return synonyms

# 定义一个函数来替换文本中的词语
def replace_synonyms(text):
    tokens = nltk.word_tokenize(text)
    for i in range(len(tokens)):
        word = tokens[i]
        synonyms = get_synonyms(word)
        if synonyms:
            # 选择第一个同义词替换原词
            replaced_word = list(synonyms)[0]
            tokens[i] = replaced_word
    return ' '.join(tokens)

# 示例文本
original_text = "The man craved a cigarette. He was addicted to nicotine."

# 替换同义词
perturbed_text = replace_synonyms(original_text)

# 打印结果
print("Original text:", original_text)
print("Perturbed text:", perturbed_text)
