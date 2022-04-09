from transformers import BertTokenizer
import sentencepiece as spm


def bert_uncased_tokenize(fin, fout, tok):
    fin = open(fin, 'r', encoding='utf-8')
    fout = open(fout, 'w', encoding='utf-8')
    # tok = BertTokenizer.from_pretrained('bert-base-uncased')
    for line in fin:
        # word_pieces = tok.tokenize(line.strip())
        word_pieces = tok.encode(line.strip(), out_type='str')
        new_line = " ".join(word_pieces)
        fout.write('{}\n'.format(new_line))


def main():
    tok = spm.SentencePieceProcessor("./ProphetNet_Multi/prophetnet_multi_dict/sentencepiece.bpe.model")
    prefix_of_vocab = [tok.id_to_piece(_id) for _id in range(20)]
    print(prefix_of_vocab)
    print("The order of current SPM is different from the `dict.txt`")

    bert_uncased_tokenize('./data/train.src', './data/tokenized_train.src', tok)
    bert_uncased_tokenize('./data/train.tgt', './data/tokenized_train.tgt', tok)
    bert_uncased_tokenize('./data/valid.src', './data/tokenized_valid.src', tok)
    bert_uncased_tokenize('./data/valid.tgt', './data/tokenized_valid.tgt', tok)
    bert_uncased_tokenize('./data/test.src', './data/tokenized_test.src', tok)
    bert_uncased_tokenize('./data/test.tgt', './data/tokenized_test.tgt', tok)


if __name__ == "__main__":
    main()

