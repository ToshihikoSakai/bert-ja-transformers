# できること
* フルスクラッチのデータからtransformersライブラリを利用してBERTの事前学習を行う
* 東北大学の日本語学習モデル(wikipedia)からさらに任意のコーパスで再事前学習を行う

# imagesの作成

```sh
docker build -t bert-ja-transformers .
```

# 事前学習を行うコーパスを用意する
1行に1文のテキストを用意する

# フルスクラッチからの事前学習
トークン化： sentencepiece
tokenizer: ALBERT(sentencepieceに対応)
model: BertForMaskedLM

```sh
docker run -it --rm -v $PWD:/work -p 8888:8888 bert-ja-transformers
python main.py
```

# 事前学習モデルからさらに任意のコーパスで再事前学習
トークン化/tokenizer： Mecab + wordpiece(事前学習モデルに準ずる)
model: BertForMaskedLM
```sh
docker run -it --rm -v $PWD:/work -p 8888:8888 bert-ja-transformers
python tohoku-bert-retrain.py
```


# 参考にしたサイト
https://qiita.com/m__k/items/6f71ab3eca64d98ec4fc
https://github.com/yoheikikuta/bert-japanese

