"""
This class is for algorithm 1 of CS5012 P2.
@author 210016568
"""
from get_data import conllu_corpus, test_corpus, train_corpus
from get_matrix import tags, get_transition, get_emission
from configurations import lang, start

train_sents = conllu_corpus(train_corpus(lang))
test_sents = conllu_corpus(test_corpus(lang))
A = get_transition(train_sents)
B = get_emission(train_sents)


def predict(tag_im1, word_i):
    global_decoding = {}
    for tag in tags:
        global_decoding[(tag_im1, tag, word_i)] = A[tag_im1].prob(tag) * B[tag].prob(word_i)
    return max(global_decoding, key=lambda k: global_decoding[k])[1]


def evaluate_algo1(sents):
    count_sents = 0
    acc_sents = 0
    count_words = 0
    err_words = 0
    for sent in sents:
        test_sentence = []
        test_tag = []
        count_sents += 1
        for token in sent:
            count_words += 1
            test_sentence.append(token['form'])
            test_tag.append(token['upos'])
        predicted_tag = [predict(start, test_sentence[0])]
        for i in range(1, len(test_sentence)):
            predicted_tag.append(predict(predicted_tag[-1], test_sentence[i]))
        equal = True
        for i in range(len(test_sentence)):
            if test_tag[i] != predicted_tag[i]:
                err_words += 1
                equal = False
        if equal:
            acc_sents += 1
    return acc_sents / count_sents, 1 - err_words / count_words


sents_acc, words_acc = evaluate_algo1(conllu_corpus(test_corpus(lang)))
print('when language = {}, the accuracy of algorithm 1 is:\nsentences acc: {:.2%}, words acc: {:.2%}'.
      format(lang, sents_acc, words_acc))
