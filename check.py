from transformers import AlbertTokenizer
from transformers import BertForMaskedLM
from transformers import pipeline
import os



def main():

    dir = os.getcwd()

    # tokenizerとmodel
    tokenizer = AlbertTokenizer.from_pretrained(dir+'/model/sentencepiece.model', keep_accents=True)
    model = BertForMaskedLM.from_pretrained(dir + '/outputBERT')

    fill_mask = pipeline(
        "fill-mask",
        model=model,
        tokenizer=tokenizer
    )

    MASK_TOKEN = tokenizer.mask_token

    # コーパスに応じた文章から穴埋めをとく

    text = "この物語の主人公は、彼《か》のバルカン地方の伝説『吸血鬼』にも比すべき、{}の悪魔である。".format(MASK_TOKEN)
    fill_mask(text)


if __name__ == '__main__':
    main()