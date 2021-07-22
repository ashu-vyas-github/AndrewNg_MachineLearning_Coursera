import numpy


def get_vocab_list():
    """
    Reads the fixed vocabulary list in vocab.txt and returns a cell array of the words
    %   vocabList = GETVOCABLIST() reads the fixed vocabulary list in vocab.txt
    %   and returns a cell array of the words in vocabList.

    :return:
    """
    vocab_list = numpy.genfromtxt("./vocab.txt", dtype=object)
    vocab_list = list(vocab_list[:, 1].astype(str))
    return vocab_list
