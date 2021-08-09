from transformers import BertConfig
from transformers import BertJapaneseTokenizer, BertModel
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

    # 東北大学BERTのtokenizer(mecab+wordpiece)を使う
    tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')

    print(tokenizer)
    # textをトークナイズ
    #text = ""
    #print(tokenizer.tokenize(text))

    # BERTモデル
    # BERT MLMのインスタンスを生成
    model = BertForMaskedLM.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
    model.to(device)

    print(model)

    # パラメータ数を表示
    print('No of parameters: ', model.num_parameters())
    
    # textを1行ずつ読み込んでトークンへ変換
    dataset = LineByLineTextDataset(
         tokenizer=tokenizer,
         file_path=dir + '/corpus/corpus.txt',
         block_size=512, # tokenizerのmax_length
         # block_sizeはtokenizerのmax_lengthっぽい
         #https://github.com/huggingface/transformers/blob/master/src/transformers/data/datasets/language_modeling.py#L114


    )

    print('No. of lines: ', len(dataset))

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
        save_steps=2000,
        save_total_limit=2,
        prediction_loss_only=True,
        logging_steps=2000,
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