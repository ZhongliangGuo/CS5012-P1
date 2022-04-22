from conllu import parse_incr

treebank = {'en': 'dataset/UD_English-EWT/en_ewt',
            'es': 'dataset/UD_Spanish-GSD/es_gsd',
            'nl': 'dataset/UD_Dutch-Alpino/nl_alpino',
            'gre': 'dataset/UD_Greek-GDT/el_gdt',
            'chn': 'dataset/UD_Chinese-GSD/zh_gsd',
            'jp': 'dataset/UD_Japanese-GSD/ja_gsd'}


def train_corpus(lang):
    return treebank[lang] + '-ud-train.conllu'


def test_corpus(lang):
    return treebank[lang] + '-ud-test.conllu'


# Remove contractions such as "isn't".
def prune_sentence(sent):
    return [token for token in sent if type(token['id']) is int]


def conllu_corpus(path):
    data_file = open(path, 'r', encoding='utf-8')
    sents = list(parse_incr(data_file))
    return [prune_sentence(sent) for sent in sents]
