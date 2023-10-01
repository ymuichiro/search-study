from janome.tokenizer import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer


# 分割したい日本語の文章のコーパス
corpus = [
    "濃厚豚骨ラーメンの魅力を徹底解剖",
    "秘密のラーメン店巡り、穴場を発見",
    "季節限定ラーメン特集、おすすめ3選",
]


def split_words(text: str) -> list[str]:
    """文章をトークナイズ"""
    # 文章を単語で分割
    tokenizer = Tokenizer()
    tokens = tokenizer.tokenize(text)

    # 分割された単語を表示
    tokenized_corpus: list[str] = []
    for token in tokens:
        part_of_speech = token.part_of_speech.split(",")[0]  # type: ignore
        surface = token.surface  # type: ignore
        if part_of_speech in ["名詞", "動詞"]:
            tokenized_corpus.append(surface)
    return tokenized_corpus


# コーパス内の文章をトークナイズしてリストに格納
tokenized_corpus = [split_words(text) for text in corpus]

# TF-IDFベクトル化器を初期化
tfidf_vectorizer = TfidfVectorizer()

# トークナイズされたコーパスをTF-IDFベクトルに変換
tfidf_matrix = tfidf_vectorizer.fit_transform(
    [" ".join(tokens) for tokens in tokenized_corpus]
)

# 各文章から重要な単語を抽出
for i, _ in enumerate(corpus):
    feature_names = tfidf_vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.getrow(i).toarray()[0]

    # TF-IDFスコアが高い上位3つの単語を抽出し、配列の index を生成 -> 抽出
    # TF = 該当文書における頻出率 × IDF = 全文書における頻出率の低さ
    important_words_indices = tfidf_scores.argsort()[::-1][:3]

    # index から単語を取得
    important_words = [feature_names[idx] for idx in important_words_indices]

    print(f"文書 {i + 1} の重要な単語:", important_words)

# qdrant で近似する文書を引っ張ってきて tf-idf -> 格納。
# index は都度更新しなければならないと思われる
