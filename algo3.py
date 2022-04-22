from get_matrix import get_emission, get_transition, tags
from configurations import start, end, lang
from get_data import test_corpus, conllu_corpus, train_corpus
from sys import float_info
from math import log, exp

min_log_prob = -float_info.max
train_sents = conllu_corpus(train_corpus(lang))
test_sents = conllu_corpus(test_corpus(lang))
A = get_transition(train_sents)
B = get_emission(train_sents)


def logsumexp(vals):
    if len(vals) == 0:
        return min_log_prob
    m = max(vals)
    if m == min_log_prob:
        return min_log_prob
    else:
        return m + log(sum([exp(val - m) for val in vals]))


def forward(sentence):
    alpha = [{}]

    # Initialize when i = 1, because input don't have <s> tag
    for tag in tags:
        alpha[0][tag] = A[start].logprob(tag) + B[tag].logprob(sentence[0])

    # i = t + 1
    # start when i = 2
    for t in range(1, len(sentence)):
        alpha.append({})
        for curr_tag in tags:
            logprob_list = []
            for prev_tag in tags:
                logprob_list.append(
                    alpha[t - 1][prev_tag] + A[prev_tag].logprob(curr_tag) + B[curr_tag].logprob(sentence[t]))
            alpha[t][curr_tag] = logsumexp(logprob_list)
    logprob_list = []
    alpha.append({})
    for tag in tags:
        logprob_list.append(alpha[-2][tag] + A[tag].logprob(end))
    alpha[-1][end] = logsumexp(logprob_list)
    alpha.insert(0, {})
    alpha[0][start] = 0
    return alpha


def backward(sentence: list):
    beta = []
    # initialize the size of beta
    for i in range(len(sentence) + 2):
        beta.append({})
    # P(end)=1, logprob(end) = 0
    beta[-1][end] = 0
    # initialize when i = len(sentence)
    for prev_tag in tags:
        beta[-2][prev_tag] = A[prev_tag].logprob(end)
    for i in range(len(sentence) - 1, 0, -1):
        for prev_tag in tags:
            logprob_list = []
            for post_tag in tags:
                logprob_list.append(
                    beta[i + 1][post_tag] + A[prev_tag].logprob(post_tag) + B[post_tag].logprob(sentence[i + 1 - 1]))
            beta[i][prev_tag] = logsumexp(logprob_list)
    logprob_list_start = []
    for tag_1 in tags:
        logprob_list_start.append(beta[1][tag_1] + A[start].logprob(tag_1) + B[tag_1].logprob(sentence[0]))
    beta[0][start] = logsumexp(logprob_list_start)
    return beta


def baum_welch(sentence):
    alpha = forward(sentence)
    beta = backward(sentence)
    predicted_tags = []
    for t in range(len(sentence)):
        tag_prob = {}
        for tag in tags:
            tag_prob[tag] = alpha[t + 1][tag] + beta[t + 1][tag]
        predicted_tags.append(max(tag_prob, key=tag_prob.get))
    return predicted_tags


def evaluate_algo3(sents):
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
        predicted_tag = baum_welch(test_sentence)
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


# sents_acc, words_acc = evaluate_algo3(conllu_corpus(test_corpus(lang)))
# print('when language = {}, the accuracy of algorithm 3 is:\nsentences acc: {:.2%}, words acc: {:.2%}'.
#       format(lang, sents_acc, words_acc))
