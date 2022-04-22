from nltk import FreqDist, WittenBellProbDist
from get_data import train_corpus, test_corpus, conllu_corpus
from configurations import lang, start, end, bins

train_sents = conllu_corpus(train_corpus(lang))
test_sents = conllu_corpus(test_corpus(lang))

tags = set()
for sent in test_sents:
    for token in sent:
        tags.add(token['upos'])
for sent in train_sents:
    for token in sent:
        tags.add(token['upos'])


def get_transition(sents):
    trans = {p: [] for p in tags}
    trans[start] = []
    for s in sents:
        trans[start].append(s[0]['upos'])
        for i, t in enumerate(s[1:]):
            trans[s[i - 1]['upos']].append(s[i]['upos'])
        trans[s[-1]['upos']].append(end)
    return smooth(trans)


def get_emission(sents):
    emission = {p: [] for p in tags}
    for s in sents:
        for t in s:
            emission[t['upos']].append(t['form'])
    return smooth(emission)


def smooth(matrix):
    new_matrix = {}
    for p in matrix.keys():
        new_matrix[p] = WittenBellProbDist(FreqDist(matrix[p]), bins=bins)
    return new_matrix
