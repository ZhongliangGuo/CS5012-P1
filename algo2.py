from get_matrix import get_emission, get_transition, tags
from configurations import start, end, lang
from get_data import test_corpus, conllu_corpus, train_corpus

train_sents = conllu_corpus(train_corpus(lang))
test_sents = conllu_corpus(test_corpus(lang))
A = get_transition(train_sents)
B = get_emission(train_sents)


def viterbi(sentence):
    V = [{}]
    path = {}

    # Initialize when i = 1, because input don't have <s> tag
    for tag in tags:
        V[0][tag] = A[start].logprob(tag) + B[tag].logprob(sentence[0])
        path[tag] = [tag]

    # i = t + 1
    for t in range(1, len(sentence)):
        V.append({})
        newpath = {}

        for curr_tag in tags:
            paths_to_curr_tag = []
            for prev_tag in tags:
                paths_to_curr_tag.append(
                    (V[t - 1][prev_tag] + A[prev_tag].logprob(curr_tag) + B[curr_tag].logprob(sentence[t]), prev_tag))
            curr_prob, prev_tag = max(paths_to_curr_tag)
            V[t][curr_tag] = curr_prob
            newpath[curr_tag] = path[prev_tag] + [curr_tag]

        # No need to keep the old paths
        path = newpath
    prob, end_state = max([(V[-1][tag] + B[tag].logprob(end), tag) for tag in tags])
    return path[end_state]


def evaluate_algo2(sents):
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
        predicted_tag = viterbi(test_sentence)
        # print for debugging
        # print(
        #     'for sentence[{}],\ntrue:tag:\n{}\npredicted tag:\n{}'.format(' '.join(test_sentence), ' '.join(test_tag),
        #                                                                   ' '.join(predicted_tag)))
        equal = True
        for i in range(len(test_sentence)):
            if test_tag[i] != predicted_tag[i]:
                err_words += 1
                equal = False
        if equal:
            acc_sents += 1
    return acc_sents / count_sents, 1 - err_words / count_words


sents_acc, words_acc = evaluate_algo2(conllu_corpus(test_corpus(lang)))
print('when language = {}, the accuracy of algorithm 2 is:\nsentences acc: {:.2%}, words acc: {:.2%}'.
      format(lang, sents_acc, words_acc))
