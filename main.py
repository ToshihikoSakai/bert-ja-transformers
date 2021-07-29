from sentencepiece import SentencePieceTrainer
from transformers import AlbertTokenizer
from transformers import BertConfig
from transformers import BertForMaskedLM
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments
from transformers import Trainer



# 事前学習用コーパスの準備
# 1行に1文章となるようなテキストを準備する
df_header = pd.read_csv('XXX.csv')
print(df_header)



# sentencepieceの学習
SentencePieceTrainer.Train(
    '--input='+dir+'corpus/corpus.txt, --model_prefix='+dir+'model/sentencepiece --character_coverage=0.9995 --vocab_size=100'
)


# ALBERTのトークナイザを定義
tokenizer = AlbertTokenizer.from_pretrained(dir+'model/sentencepiece.model', keep_accents=True)

# textをトークナイズ
text = "吾輩は猫である。名前はまだ無い。"
print(tokenizer.tokenize(text))


# BERTモデルのconfigを設定
# BERTconfigを定義
config = BertConfig(vocab_size=32003, num_hidden_layers=12, intermediate_size=768, num_attention_heads=12)

# BERT MLMのインスタンスを生成
model = BertForMaskedLM(config)

# パラメータ数を表示
print('No of parameters: ', model.num_parameters())


# 事前学習用のデータセットを準備

# textを1行ずつ読み込んでトークンへ変換
dataset = LineByLineTextDataset(
     tokenizer=tokenizer,
     file_path=dir + 'corpus/corpus.txt',
     block_size=256, # tokenizerのmax_length
)

# データセットからサンプルのリストを受け取り、それらをテンソルの辞書としてバッチに照合するための関数
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=True,
    mlm_probability= 0.15
)


training_args = TrainingArguments(
    output_dir= drive_dir + 'outputBERT/',
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_device_train_batch_size=32,
    save_steps=10000,
    save_total_limit=2,
    prediction_loss_only=True
)

# trainerインスタンスの生成
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset
)

#　学習
trainer.train()

# 学習したモデルの保存
trainer.save_model(dir + 'outputBERT/')

# 言語モデルの確認
# tokenizerとmodel
tokenizer = AlbertTokenizer.from_pretrained(drive_dir+'model/sentencepiece.model', keep_accents=True)
model = BertForMaskedLM.from_pretrained(drive_dir + 'outputBERT')

fill_mask = pipeline(
    "fill-mask",
    model=model,
    tokenizer=tokenizer
)

MASK_TOKEN = tokenizer.mask_token

# コーパスに応じた文章から穴埋めをとく

text = "XXX{}XXX".format(MASK_TOKEN)
fill_mask(text)
