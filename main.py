import spacy
import functools

nlp = spacy.load("en_core_web_sm")

def translate(text):
    """ Translate text into Yodish """
    doc = nlp(text)
    return translate_sents(doc.sents)

# word.py

class Word:
    def __init__(self, text, tag):
        self.text = text.lower()
        self.tag = tag
        self.expand_contractions()
        self.apply_capitalization()
        
    def __str__(self):
        return self.text

    def __repr__(self):
        return self.text

    def apply_capitalization(self):
        if self.text == 'i' or self.tag == 'NNP':
            self.text = capitalize(self.text)

    def expand_contractions(self):
        """ Like contractions, Yoda does not. """
        if self.text in contractions and self.tag != "POS":
            self.text = contractions[self.text]

def capitalize(s):
    return s[0].upper() + s[1:]

contractions = {
    '\'ll' : 'will',
    '\'d' : 'would',
    '\'ve': 'have',
    '\'re': 'are',
    'n\'t' : 'not',
    '\'s' : 'is',
    '\'m' : 'am',
};

# text.py

punctuation = [u',', u'.', u';', u'?', u'!', u':']
stop_punctuation = [ u'.', u'!', u'?' ]

def translate_sents(text):
    """ Applies translation rules to some spacy-processed text """
    result = [
        [ apply_yodish_grammar(clause_chunk)
          for clause_chunk in split_clauses(clause)
        ]
        for clause in text
    ]
    return serialize(flatten(result))

def flatten(text):
    """Turns text (which contains clauses which contain clause chunks which contain
words) into a flattened list of words
    """
    return [
        word
        for clause in text
        for clause_chunk in clause
        for word in clause_chunk
    ]

def split_clauses(clause):
    """ Attempt to roughly split spacy-interpreted clauses into a "clause chunk"
(targets for our yodish-translation rules """
    output = []
    curr = []
    def save_to_output(chunk):
        if chunk != []:
            output.append(chunk)
        
    for token in clause:
        if token.dep_ == u'cc' or (token.dep_ == u'punct' and token.text in punctuation):
            save_to_output(curr)
            save_to_output([ Word(token.text, token.tag_)])
            curr = []
        else:
            curr.append(Word(token.text, token.tag_))

    save_to_output(curr)
    return output


def serialize(words):
    """ Turn a list of words into a string sentence with "standard" formatting and capitalization conventions
    """
    string = []
    for (i, word) in enumerate(words):
        if (i > 0 and word.text in punctuation) or word.tag == "POS":
            string[-1] = string[-1] + word.text
        elif (i > 0 and words[i - 1].text in stop_punctuation):
            string.append(capitalize(word.text))
        else:
            string.append(word.text)
            
    return capitalize(" ".join(string))

# rules.py

def index_tag_seq(words, seq):
    """ Return index of first occurrence of seq in words (fuzzy) """
    if len(seq) > len(words):
        return -1
    
    nouns = [ 'NN', 'NNS', 'NNP' ]
    seq = [ ('NN' if tag in nouns else tag) for tag in seq ]
    tags = [ ('NN' if w.tag in nouns else w.tag) for w in words ]

    for i in range(len(tags)):
        if tags[i:i+len(seq)] == seq:
            return i

    return -1

def move_tag_seq(words, seq, dest, punc=None):
    """ If seq present (order matters), move words to dest.
    Prepend (for 'end') or append (for 'start') with punctuation if required. """
    seq_start = index_tag_seq(words, seq)
    if seq_start > -1:
        move_words = words[seq_start:seq_start+len(seq)]
        rest = words[:seq_start] + words[seq_start+len(seq):]
        punc = [ punc ] if punc else []
        if dest == 'start':
            words = move_words + rest + punc 
        if dest == 'end':
            words = rest + punc + move_words
        return words
    return None

def replace_tag_seq(words, seq1, seq2):
    """ Move/change words matching tag sequence 1 to match sequence 2. 
        Weird things may happen if tags are duplicated in seq1/seq2 """
    seq_start = index_tag_seq(words, seq1)
    if seq_start > -1:
        pre = words[:seq_start]
        post = words[seq_start+len(seq1):]
        tag_to_word = dict([ (word.tag, word) for word in words[seq_start:seq_start + len(seq1)] ])
        new = filter(lambda x : x is not None, [ tag_to_word[x] if x in tag_to_word else None for x in seq2 ])
        return pre + list(new) + post
    return None

def rule_prp_vbp(words):
    """ You are conflicted. -> Conflicted, you are. """
    return move_tag_seq(words, ['PRP', 'VBP'], 'end', Word(',', 'punct'))


def rule_rb_jjr(words):
    """ I sense much anger in him. -> Much anger I sense in him. """
    return move_tag_seq(words, ['RB', 'JJR'], 'start')


def rule_vb_prp_nn(words):
    """ Put your weapons away. -> Away put your weapons. """
    if index_tag_seq(words, ['VB', 'PRP$', 'NNS', 'RB']) > -1:
        return move_tag_seq(words, ['VB', 'PRP$', 'NNS'], 'end')
    return None


def rule_dt_vbz(words):
    """ This is my home. -> My home this is. """
    return move_tag_seq(words, ['DT', 'VBZ'], 'end')


def rule_nnp_vbz_rb_vb(words):
    """ Size does not matter. -> Size matters not. 
    Conversion of VB to VBZ is blunt at best (adding 's'). """
    original_len = len(words)
    words = replace_tag_seq(
        words,
        ['NNP','VBZ','RB','VB'],
        ['NNP','VB','RB']
    )
    if words is not None:
        if len(words) < original_len:
            i = index_tag_seq(words, ['NNP', 'VB', 'RB'])
            words[i+1].text += 's'
            words[i+1].tag = 'VBZ'
    return words
    
def apply_yodish_grammar(clause):
    def apply_rule(inp, r):
        applied = r(inp)
        return (applied if applied else inp)

    rules = [
        rule_prp_vbp,
        rule_rb_jjr,
        rule_vb_prp_nn,
        rule_dt_vbz,
        rule_nnp_vbz_rb_vb,
    ]
    return functools.reduce(apply_rule, rules, clause)

print(translate("I sense much anger in him."))
