from transformers import BertTokenizer
import sentencepiece as spm

REPL_SEP_LIST = []
NEW_SEP_LIST = []


def find_sub_idx(tgt_list, repl_list, start=0):
    repl_length = len(repl_list)
    for idx in range(start, len(tgt_list)):
        if tgt_list[idx:idx+repl_length] == repl_list:
            return idx, idx+repl_length


def replace_sub(tgt_list, repl_list, new_list):
    new_length = len(new_list)
    idx = 0
    for start, end in iter(lambda: find_sub_idx(tgt_list, repl_list, idx), None):
        tgt_list[start:end] = new_list
        idx = start + new_length


def bert_uncased_tokenize(fin, fout, tok):
    fin = open(fin, 'r', encoding='utf-8')
    fout = open(fout, 'w', encoding='utf-8')
    # tok = BertTokenizer.from_pretrained('bert-base-uncased')
    for line in fin:
        # word_pieces = tok.tokenize(line.strip())
        word_pieces = tok.encode(line.strip(), out_type='str')
        replace_sub(word_pieces, REPL_SEP_LIST, NEW_SEP_LIST)
        new_line = " ".join(word_pieces)
        fout.write('{}\n'.format(new_line))


def main():
    tok = spm.SentencePieceProcessor("./ProphetNet_Multi/prophetnet_multi_dict/sentencepiece.bpe.model")
    prefix_of_vocab = [tok.id_to_piece(_id) for _id in range(20)]
    print(prefix_of_vocab)
    print("The order of current SPM is different from the `dict.txt`")

    bert_uncased_tokenize('./data/train.src', './data/tokenized_train.src', tok)
    bert_uncased_tokenize('./data/train.tgt', './data/tokenized_train.tgt', tok)
    bert_uncased_tokenize('./data/val.src', './data/tokenized_valid.src', tok)
    bert_uncased_tokenize('./data/val.tgt', './data/tokenized_valid.tgt', tok)
    bert_uncased_tokenize('./data/test.src', './data/tokenized_test.src', tok)
    bert_uncased_tokenize('./data/test.tgt', './data/tokenized_test.tgt', tok)


if __name__ == "__main__":
    main()

