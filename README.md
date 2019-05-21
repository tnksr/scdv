
# SCDV

- **S**parse      疎な
- **C**omposite   合成された
- **D**ocument    文書の
- **V**ectors     ベクトル

分類などのタスクにおいて、テキストのベクトル化は必須。

よくあるword2vecを用いた文書のベクトル化は単純な単語の分散表現の足し合わせが基本だが、  
SCDVでは、各単語のトピックや文書における影響度なども考慮している。

SkipGramやBoWVよりも文書分類のタスクにおいて精度がよかった。

論文：https://www.aclweb.org/anthology/D17-1069

## 論文のポイント

- syntax and semantics
    - 文法的、意味的に単語を学習している（既存のword2vecと同様）
- latent topic model
    - latent: 潜在的
    - トピックモデル（のようなもの）も学習
    - カテゴリによって異なる単語の出現頻度を考慮
    - 文書のカテゴリによって持つ意味が変化する単語にも対応
        - "apple"は食べ物の話題なのかIT業界の話題なのかで意味合いが変わってくる
- sparce
    - 処理の高速化

#### 主張
    
- 文書ベクトルは単語ベクトルより高次元であるべき
    - 単なる足し合わせ（単語の集合）以上の意味を持つ（トピックのような）
- 単語によって重要性(importance)や特殊性(distinctiveness)がある
    - カテゴリによって意味（重要性）が異なる単語もある

## 手法

1. 単語ベクトル $wv_i$ を求める。
2. GMMで単語ベクトル $wv_i$ をクラスタリングする。
3. 各単語ベクトルが各クラスタに属する確率 $P(c_k|wv_i)$ を求める。
4. 単語クラスタベクトル $wcv_{ik} := wv_i \times P(c_k|wv_i)$を求める。
5. 単語のIDF値 ${\rm idf}(w_i)$ を求める。
6. 単語トピックベクトル $wtv_i := {\rm idf} (w_i) \times \oplus_{k=1}^{K} {wcv}_{ik}$を求める。
7. 文書中の各単語の単語トピックベクトルを足し合わせ、文書ベクトルを得る。
8. スパース化

<img border="0" src="./docs/image/wtv.png" height="3">

<img border="0" src="./docs/image/wtv_image.png" height="3">

## 実装

### 0. コーパスの作成
- 前処理
- 分かち書き

```python
corpus = Corpus(min_length)
corpus.build(input_file, tokenizer=tokenizer)
```

### 1. 単語ベクトルを求める

$$
wv_i (w_i \in V)
$$

- 単語の分散表現を得る
- 全体の文書から学習済みのものを用いる


```python
word2vec = word2vec.Word2Vec.load('./word2vec.model')
```

### 2. GMMで単語ベクトルをクラスタリングする。

$$
c_k ( k \in K )
$$

#### アイデア
- 単語はトピックを持つ
- 文書は複数のトピックから成る

```python
gm = GaussianMixture(n_components=cluster_size, max_iter=max_iter)
gm = gm.fit(word2vec.wv.vectors)
```

### 3. 各単語ベクトルが各クラスタに属する確率を求める。

$$
P(c_k|w_i)
$$

#### トピックモデルとは
- トピックごとに単語の出現頻度分布を想定
- 文書内の各単語はトピックが持つ確率分布に従って出現すると仮定
    - 「サッカー」という単語は「スポーツ」の文書に出現しやすい
    - 「日本」という単語は「経済」の文書に出現しやすい


#### アイデア
- 各単語ベクトルが各クラスタに属する確率 $\fallingdotseq$ 各単語の各トピックでの出現確率 $\fallingdotseq$ 単語の持つトピック


```python
embedding = np.array([word2vec.wv[word] for word in id_to_word])
probability = gm.predict_proba(embedding)
```

### 4. 単語クラスタベクトルを求める。

$$
wcv_{ik} = wv_i \times P(c_k|w_i)
$$

#### アイデア
- 各単語はトピックを持つ（複数のトピックを持つ）
- 各トピックでの単語の意味・重要性


```python
vocab_size, embedd_size = embedding.shape
e = embedding.reshape(vocab_size, 1, embedd_size)
p = probability.reshape(vocab_size, cluster_size, 1)
wcv = e * p
```

### 5. 単語のIDF値を求める。


$$
{\rm idf}(w_i) = \log{\frac{|D|}{|\{d: d \ni w_i|\}}}
$$

#### アイデア
- idfが大きい単語のほうが文書への影響力は大きい


```python
tokenizer = ' '
tfv = TfidfVectorizer(dtype=np.float32, max_df=con.MAX_DF, min_df=con.MIN_DF,
                      stop_words=con.STOP_WORDS, smooth_idf=con.SMOOTH_IDF)
documents = [tokenizer.join(text) for text in corpus.documents]
_ = tfv.fit_transform(documents)
```


```python
featurenames = tfv.get_feature_names()
idf = {}
for name, feature in zip(featurenames, tfv._tfidf.idf_):
    idf[name] = feature
idf = np.array([idf[word] if idf.get(word) else 0.0 for word in id_to_word])
```

### 6. 単語トピックベクトルを求める。

$$
wtv_i = {\rm idf}(w_i) \times \oplus_{k=1}^{K} wcv_{ik}
$$


#### アイデア
- 文書への影響力が大きい単語は、その単語のトピックも影響力が大きくなる
    - ${\rm idf} \times wcv_i$



```python
wtv = wcv * idf.reshape(vocab_size, 1, 1)
```

### 7. 文書中の各単語の単語トピックベクトルを足し合わせ、文書ベクトルを得る。

#### アイデア
- 文書に多く含まれるトピックほど影響力が大きくなる。
    - スポーツの話題（スポーツに関係する単語が多く含まれる文書）ではスポーツに関係する単語の影響力大きく、政治に関係する単語の影響力は小さくなる。


```python
get_vector = lambda text: [wtv[word_to_id[word]] for word in text if word_to_id.get(word)]
document_vectors = [get_vector(document) for document in corpus.documents]
```

### 8. スパース化

- 絶対値が小さい要素はゼロにする


```python
sparsity_percentage = 0.7
```


```python
abs_ave_max = lambda array : np.abs(np.average(np.max(array, axis=1)))
threshold = sparsity_percentage * (abs_ave_max(document_vectors) + abs_ave_max(-document_vectors))/2
document_vectors[document_vectors < threshold] = 0
```
