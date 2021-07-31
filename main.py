from sentencepiece import SentencePieceTrainer
from transformers import AlbertTokenizer
from transformers import BertConfig
from transformers import BertForMaskedLM
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments
from transformers import Trainer
from transformers import pipeline
import pandas as pd
import os
import torch

def main():

    cuda_yes = torch.cuda.is_available()
    print('Cuda is available?', cuda_yes)
    device = torch.device("cuda:0" if cuda_yes else "cpu")
    print('Device:', device)


    dir = os.getcwd()

    # 事前学習用コーパスの準備
    # 1行に1文章となるようなテキストを準備する
    #df_header = pd.read_csv('XXX.csv')
    #print(df_header)

    # vocab_sizeの設定
    vocab_size = 18000

    # Tokenization
    # sentencepieceの学習
    SentencePieceTrainer.Train(
        '--input='+dir+'/corpus/corpus.txt, --model_prefix='+dir+'/model/sentencepiece --character_coverage=0.9995 --vocab_size={}'.format(vocab_size)
    )

    # sentencepieceのパラメータ
    # https://github.com/google/sentencepiece#train-sentencepiece-model
    # training options
    # https://github.com/google/sentencepiece/blob/master/doc/options.md


    # sentencepieceを使ったTokenizerは現時点では以下。
    # >All transformers models in the library that use SentencePiece use it 
    # in combination with unigram. Examples of models using SentencePiece are ALBERT, XLNet, Marian, and T5.
    # https://huggingface.co/transformers/tokenizer_summary.html
    # ALBERTのトークナイザを定義
    tokenizer = AlbertTokenizer.from_pretrained(dir+'/model/sentencepiece.model', keep_accents=True)

    print(tokenizer)
    # textをトークナイズ
    #text = ""
    #print(tokenizer.tokenize(text))

    # BERTモデルのconfigを設定
    # BERTconfigを定義
    # vocab_sizeはspecial_tokensの数を加算する必要がある
    config = BertConfig(vocab_size=18007, num_hidden_layers=12, intermediate_size=768, num_attention_heads=12)

    # BERT MLMのインスタンスを生成
    model = BertForMaskedLM(config)
    model.to(device)

    print(model)

    # パラメータ数を表示
    print('No of parameters: ', model.num_parameters())
    
    # textを1行ずつ読み込んでトークンへ変換
    dataset = LineByLineTextDataset(
         tokenizer=tokenizer,
         file_path=dir + '/corpus/corpus.txt',
         block_size=256, # tokenizerのmax_length
    )

    # データセットからサンプルのリストを受け取り、それらをテンソルの辞書としてバッチに照合するための関数
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=True,
        mlm_probability= 0.15
    )

    # 事前学習のパラメータを定義
    training_args = TrainingArguments(
        output_dir= dir + '/outputBERT/',
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
    trainer.save_model(dir + '/outputBERT/')





if __name__ == '__main__':
    main()