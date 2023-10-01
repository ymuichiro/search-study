from janome.tokenizer import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer


def text_optimize(text: str) -> str:
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
    return " ".join(tokenized_corpus)


def execute_important_words(corpus: list[str]):
    # TF-IDFベクトル化器を初期化
    tfidf_vectorizer = TfidfVectorizer()

    # TF-IDFベクトルに変換
    sparse = tfidf_vectorizer.fit_transform(corpus)

    # 各文章から重要な単語を抽出
    for i, _ in enumerate(corpus):
        feature_names = tfidf_vectorizer.get_feature_names_out()
        tfidf_scores = sparse.getrow(i).toarray()[0]

        # TF-IDFスコアが高い上位3つの単語を抽出し、配列の index を生成 -> 抽出
        # TF = 該当文書における頻出率 × IDF = 全文書における頻出率の低さ
        important_words_indices = tfidf_scores.argsort()[::-1][:5]
        not_import_words_indices = tfidf_scores.argsort()[:5]

        # index から単語を取得
        important_words = [feature_names[idx] for idx in important_words_indices]
        not_import_words = [feature_names[idx] for idx in not_import_words_indices]

        print(f"文書 {i + 1} の重要な単語:", important_words, "そうでない単語:", not_import_words)


# 分割したい日本語の文章のコーパス
CORPUS = [
    "濃厚豚骨ラーメンの魅力を徹底解剖",
    "秘密のラーメン店巡り、穴場を発見",
    "季節限定ラーメン特集、おすすめ3選",
]

# コーパス内の文章をトークナイズしてリストに格納
tokenized_corpus: list[str] = [text_optimize(text) for text in CORPUS]
execute_important_words(tokenized_corpus)

# qdrant で近似する文書を引っ張ってきて tf-idf -> 格納。
# index は都度更新しなければならないと思われる
# index = tokenize された文書タイトル, not importance words, importance words
# タイトルに文書を表す言葉を入れてください
# chunk テキスト自体にこれを適用するべき？ chunk 化されたテキストの中だけでこれを行うとどうなるか？
