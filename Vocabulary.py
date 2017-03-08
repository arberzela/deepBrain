import numpy as np
from itertools import islice


FILE_NAME_TRAIN = "train.txt" #change accordingly
FILE_NAME_VALID = "valid.txt" #change accordingly
LARGEST_UINT32 = 4294967295


def text_to_pairs(text, random_gen, half_window_size=2, nsamples_per_word=1):
    npairs = sum([2 * len(doc) * half_window_size * nsamples_per_word for doc in text])
    pairs = np.empty((npairs, 4), dtype=np.uint32)
    randids = random_gen(npairs)
    next_pair = 0
    for doc in text:
        cdoc = doc
        doc_len = len(cdoc)
        for i in range(doc_len):
            if cdoc[i] == LARGEST_UINT32:
                continue
            for j in range(i + 1, min(i + half_window_size + 1, doc_len)):
                if cdoc[j] == LARGEST_UINT32:
                    continue
                for k in range(nsamples_per_word):
                    pairs[next_pair, 0] = cdoc[i]
                    pairs[next_pair, 1] = cdoc[j]
                    pairs[next_pair, 2] = cdoc[i]
                    pairs[next_pair, 3] = randids[next_pair]
                    next_pair += 1

    return np.ascontiguousarray(pairs[:next_pair, :])


class Vocabulary(object):
    """
    A vocabulary contains information about unique words/tokens and chars of a corpora.
    It can build a wrd2idx,idx2wrd,char2idx,idx2char dict by reading a textfile and can hash words to the corresponding idx-array.

    The most important dicts are:
        * dict_wrds2idx : Maps words to idx
        * dict_idx2wrds : Maps idx to wrds

        examples: dict_wrds2idx['are'] = 1

    """

    def __init__(self, unk_wrd='<unkown>'):
        """
        Creates a vocabulary.

        :param unk_wrd: Words with are unkown (not in dicts) are replaced with this char.

        """

        # Setting the attributes
        self.unk_wrd = unk_wrd

        # init wrd_dicts with specified unknown wrd
        self.dict_wrds2idx = {}
        self.dict_idx2wrds = {}
        self.wrdlen = {}

        self.dict_wrds2idx[self.unk_wrd] = 0
        self.dict_idx2wrds[0] = self.unk_wrd

        self.wrdlen[0] = len(self.unk_wrd)
        self.max_word_len = -1

    def hash_file_linewise(self, filename, max_lines=None, num_valid_lines=None, extend_wrd_dict=True,
                           eos_symbol=' <eos> ', separate_sentences=True, wrd_to_lowercase=True):

        """
        Hashes a file linewise, i.e. hash the string representation of words/tokens to a idx representation (e.g. "That means that" to [1,2,1]).
        Each line represents a sentence, so that the text file has to be in that form.


        :param filename: Path to your file

        :param max_lines: Maximal number of lines or sentences, which should be hashed

        :param num_valid_lines: The first n lines will be used as valid data.
                                When this is not None, than the return value "ret_data_hashed" is a list of train and valid data.

        :param extend_wrd_dict: When specified the dictionaries will not be extended. Unknown words will be replace with unknown symbols.
                                'False' can be used, when you want to hash your test data after you trained a language model.

        :param eos_symbol: Part of Speach Tag to indicate the end of sentence (EOS), If set set 'None' then no EOS will be added.

        :param separate_sentences: Determines if the data should be hashed to one large list (seperate_sentences = False)
                                    or if for each sentence a new list should be generated. Can be used to manually split
                                    the data into valid and train set e.g. for K-cross-Validation.

                                    Examples
                                    --------------------------------------
                                    sentence1 = 'I am hungry', sentence2 = 'We should go mensa', dict_wrd2idx['<eos>'] = 4
                                    seperate_sentences = True : ret_data_hashed = [[1,2,3,4],[5,6,7,8,9]]
                                    seperate_sentences = False : ret_data_hashed = [1,2,3,4,5,6,7,8,9]

                                    Attention if seperate_sentences = True:
                                    -----------------------------------------
                                    The length of sentences could vary from sentence to sentence. So you need to flat
                                    your data into one big list before feeding your neuronal network.

        :param wrd_to_lowercase: Should all words only be treated as lowercase?

        :return:ret_data_hashed : Depending on seperate_sentences and num_valid_lines the hashed file is returned

        :return:ret_dict_idx : dict_wrd2idx returned

        :return ret_dict : dict_idx2wrd is returned
        """

        with open(filename, encoding='utf-8') as f:

            data_hashed_train = []
            data_hashed_valid = []
            i = 0

            # Iterate over all lines
            for line in f:

                # Check if max line is reached
                if (max_lines is not None):
                    if (i >= max_lines):
                        break

                # hash sentence and extend vocabs if necessary
                hashed_sentence = self.text2idx(line, extend_wrd_dict, eos_symbol=eos_symbol,
                                                wrd_to_lowercase=wrd_to_lowercase)

                if separate_sentences:
                    hashed_sentence = [hashed_sentence]

                # adding the hashed line either to valid or to train data
                if (num_valid_lines is not None and i < num_valid_lines):
                    data_hashed_valid += hashed_sentence

                else:
                    data_hashed_train += hashed_sentence

                i += 1

            # If we do not separate sentences we create a numpy array
            if not separate_sentences:
                data_hashed_train = np.asarray(data_hashed_train, dtype='uint32')
                data_hashed_valid = np.asarray(data_hashed_valid, dtype='uint32')

            ret_data_hashed = data_hashed_train

            # Adding valid data to return data, if necessary
            if num_valid_lines is not None:
                ret_data_hashed = [ret_data_hashed] + [data_hashed_valid]

            # Saving the last hased data to vocab
            self.last_data_hashed = ret_data_hashed

            return ret_data_hashed, self.dict_wrds2idx, self.dict_idx2wrds

    def text2idx(self, text, extend_wrd_dict=True, eos_symbol=' <eos> ',
                 wrd_to_lowercase=True):
        """
        Hashes a given string to the idx representation.

        :param text: The string which should be hashed

        :param extend_wrd_dict: When specified the dictionaries will not be extended. Unknown words will be replace with unknown symbols.
                                'False' can be used, when you want to hash your test data after you trained a language model.

        :param eos_symbol: Part of Speach Tag to indicate the end of a sentence

        :return: data_hashed: The idx representation of the given text

        :param wrd_to_lowercase: Should all words only be treated as lowercase?
         """

        # Adding EOS symbol if necessary
        if eos_symbol is not None:
            text = text.replace('\n', eos_symbol).split()
        else:
            text = text.replace('\n', ' ').split()

        # Retrieve the last positions
        wrd_idx = len(self.dict_wrds2idx)

        # Init data_hashed
        data_hashed = []

        # Hash each word in text and maybe extend the dicts
        for wrd in text:

            # we only consider lowercase words, if necessary
            if wrd_to_lowercase:
                wrd = wrd.lower()

            # extending word_dict
            if (extend_wrd_dict):

                # Adding wrd to wrd dicts if previouly unknown
                if wrd not in self.dict_wrds2idx:
                    self.dict_wrds2idx[wrd] = wrd_idx
                    self.dict_idx2wrds[wrd_idx] = wrd
                    self.wrdlen[wrd_idx] = len(wrd)

                    wrd_idx += 1

            # Replace word with unkown symbol, if it is unknown
            if wrd not in self.dict_wrds2idx:
                data_hashed += [self.dict_wrds2idx[self.unk_wrd]]
            else:
                data_hashed += [self.dict_wrds2idx[wrd]]

        return data_hashed

    def random_ids(self, num):
        return np.random.randint(0, len(self.dict_wrds2idx), size=num).astype(np.uint32)

    def iter_pairs(self, fin, batch_size=10, nsamples=2, window=5):

        documents = iter(fin)
        batch = list(islice(documents, batch_size))
        while len(batch) > 0:

            pairs = text_to_pairs(batch, self.random_ids,
                                  nsamples_per_word=nsamples,
                                  half_window_size=window)
            yield pairs
            batch = list(islice(documents, batch_size))


vocab = Vocabulary()
print("Load Data")
train_data, dict, dictwrds = vocab.hash_file_linewise(FILE_NAME_TRAIN, max_lines=100)
valid_data, dict, dictwrds = vocab.hash_file_linewise(FILE_NAME_VALID, max_lines=100)
train_data = np.asarray(train_data)
vocab_size = len(dict)
print("data loaded")

print("-" * 80)
print("Vocab size: ", vocab_size)
print("Train size: ", len(train_data))
print("Valid size: ", len(valid_data))
print("Data shapes")
print("-" * 80)
