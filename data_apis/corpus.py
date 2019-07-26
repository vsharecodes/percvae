from collections import Counter
import numpy as np
import nltk


class ConvAI2DialogCorpus(object):
    dialog_act_id = 0
    sentiment_id = 1
    liwc_id = 2

    def __init__(self, corpus_path, max_vocab_cnt=20000, word2vec=None, word2vec_dim=None, vocab_files=None,
                 idf_files=None):
        self._path = corpus_path
        self.word_vec_path = word2vec
        self.word2vec_dim = word2vec_dim
        self.word2vec = None
        self.vocab = None
        self.dialog_id = 0
        self.meta_id = 1
        self.utt_id = 2
        self.persona_id = 3
        self.persona_word_id = 4
        self.sil_utt = ["<s>", "<sil>", "</s>"]
        self.persona_precursor = ["my favorite", "i love", "i like", "i am", "i'm ", "i can", "i enjoy", "i watch",
                                  "i work", "i have", "i've", "i had", "i live", "i do", "i hate", "i believe",
                                  "i have"]
        data = self.reading_convai2_corpus(self._path)
        self.train_corpus = self.process(data["train"])
        self.valid_corpus = self.process(data["valid"])
        self.test_corpus = self.process(data["test"])
        self.build_vocab(max_vocab_cnt, vocab_files, idf_files)
        self.load_word2vec()
        print("Done loading corpus")

    def reading_convai2_corpus(self, path):
        def _read_persona_and_dialogue(name, corpus):
            load_corpus = []
            utts = []
            persona = []
            persona_word = ''
            count = 0
            for l in corpus:
                count += 1
                l = l.strip()
                if l.split(' ')[0] == '1' and count > 1:
                    segments = {'utts': utts, 'persona': persona, 'persona_word': ['<s>' + persona_word + ' </s>']}
                    load_corpus.append(segments)
                    utts = []
                    per = l.split('your persona: ')[1].strip(' .').lower().strip()
                    persona = ['</s>'] + [per + ' </s>']
                    persona_word = (per + ' ')
                else:
                    if 'your persona:' in l:
                        per = l.split('your persona: ')[1].strip(' .').lower().strip()
                        persona.append(per + ' </s>')
                        persona_word += (per + ' ')
                    else:
                        if name == 'Null':
                            utt = (' '.join(l.split(' ')[1:])).split('\t')
                            utts.append(('A', utt[0] + ' ' + persona[np.random.randint(len(persona))],
                                         ['None', [0.0, 0.0, 0.0, 0.0]]))
                            utts.append(('B', utt[1] + ' ' + persona[np.random.randint(len(persona))],
                                         ['None', [0.0, 0.0, 0.0, 0.0]]))
                        else:
                            utt = (' '.join(l.split(' ')[1:])).split('\t')
                            utts.append(('A', utt[0], ['None', [0.0, 0.0, 0.0, 0.0]]))
                            utts.append(('B', utt[1], ['None', [0.0, 0.0, 0.0, 0.0]]))
            return load_corpus

        train_corpus = _read_persona_and_dialogue('Train', open(path + 'train.txt'))
        valid_corpus = _read_persona_and_dialogue('Valid', open(path + 'valid.txt'))
        test_corpus = _read_persona_and_dialogue('Test', open(path + 'test.txt'))

        return {'train': train_corpus, 'valid': valid_corpus, 'test': test_corpus}

    def process(self, data):
        new_dialog = []
        new_meta = []
        new_utts = []
        all_lens = []
        new_persona = []
        new_persona_word = []

        for l in data:
            lower_utts = [(caller, ["<s>"] + nltk.WordPunctTokenizer().tokenize(utt.lower()) + ["</s>"], feat)
                          for caller, utt, feat in l["utts"]]
            all_lens.extend([len(u) for c, u, f in lower_utts])
            vec_a_meta = [0, 0] + [0, 0]
            vec_b_meta = [0, 0] + [0, 0]
            meta = (vec_a_meta, vec_b_meta, 'NULL')
            dialog = [(utt, int(caller == "A"), feat) for caller, utt, feat in lower_utts]
            new_utts.extend([utt for caller, utt, feat in lower_utts])
            new_dialog.append(dialog)
            new_meta.append(meta)
            new_persona.append([(p.split(' ')) for p in l['persona']])
            new_persona_word.append(l['persona_word'])

        print("Max utt len %d, mean utt len %.2f" % (np.max(all_lens), float(np.mean(all_lens))))
        return new_dialog, new_meta, new_utts, new_persona, new_persona_word

    def build_vocab(self, max_vocab_cnt, vocab_files=None, idf_files=None):

        self.vocab = []
        self.rev_vocab = []
        if vocab_files is None:
            all_words = []
            persona_words = []
            for tokens in self.train_corpus[self.utt_id]:
                all_words.extend(tokens)
            for persona in self.train_corpus[self.persona_id]:
                for p in persona:
                    all_words.extend(p)
            for persona in self.train_corpus[self.persona_id]:
                for p in persona:
                    persona_words.extend(p)

            vocab_count = Counter(all_words).most_common()
            raw_vocab_size = len(vocab_count)
            vocab_count = vocab_count[0:max_vocab_cnt]

            persona_vocab_count = Counter(persona_words).most_common()

            all_vocab_set = set([t for t, cnt in vocab_count]) - set(["<s>", "</s>"])
            persona_vocab_set = set([t for t, cnt in persona_vocab_count if cnt <= -1]) - set(["<s>", "</s>"])
            normal_vocab = ["<pad>", "<unk>"] + ["<s>", "</s>"] + ["<sentinel>"] + list(
                all_vocab_set - persona_vocab_set)
            persona_vocab = list(persona_vocab_set)
            self.gen_vocab_size = len(normal_vocab)
            self.copy_vocab_size = len(persona_vocab)

            # create vocabulary list sorted by count
            print("Build corpus with raw vocab size %d, vocab size %d, gen vocab size %d, copy vocab size %d ."
                  % (raw_vocab_size, self.gen_vocab_size + self.copy_vocab_size, self.gen_vocab_size,
                     self.copy_vocab_size))

            self.vocab = normal_vocab + persona_vocab
            self.rev_vocab = {t: idx for idx, t in enumerate(self.vocab)}
            self.unk_id = self.rev_vocab["<unk>"]
            print("vocab unk_id is %d, pad_id is %d" % (self.unk_id, self.rev_vocab["<pad>"]))
            print("vocab <s> is %d, </s> is %d" % (self.rev_vocab["<s>"], self.rev_vocab["</s>"]))

        else:
            with open(self._path + vocab_files, 'r') as vocab_f:
                for vocab in vocab_f:
                    vocab = vocab.strip()
                    self.vocab.append(vocab)
            self.rev_vocab = {t: idx for idx, t in enumerate(self.vocab)}
            self.unk_id = self.rev_vocab["<unk>"]
            self.gen_vocab_size = len(self.vocab)
            self.copy_vocab_size = 0
            print("Load corpus from %s with vocab size %d ." % (vocab_files, len(self.vocab)))
            print("vocab unk_id is %d, pad_id is %d" % (self.unk_id, self.rev_vocab["<pad>"]))
            print("vocab <s> is %d, </s> is %d" % (self.rev_vocab["<s>"], self.rev_vocab["</s>"]))

        self.idf = {}
        self.index2idf = [1.0 for _ in range(len(self.vocab))]
        if idf_files is None:
            for vocab in self.vocab:
                self.idf[vocab] = 1.0
            print("All words' IDF are set to 1.0, as idf_files == None")
        else:
            with open(self._path + idf_files, 'r') as idf_f:
                for i, line in enumerate(idf_f):
                    line = line.strip().split('\t')
                    vocab = line[0]
                    idf = float(line[1])
                    self.idf[vocab] = idf
                    self.index2idf[i] = idf
            print("Load words' IDF from %s with size %d ." % (idf_files, len(self.idf)))

        # create topic vocab
        all_topics = []
        for a, b, topic in self.train_corpus[self.meta_id]:
            if topic is not None:
                all_topics.append(topic)
        self.topic_vocab = [t for t, cnt in Counter(all_topics).most_common()]
        self.rev_topic_vocab = {t: idx for idx, t in enumerate(self.topic_vocab)}

        # get dialog act labels
        all_dialog_acts = []
        for dialog in self.train_corpus[self.dialog_id]:
            all_dialog_acts.extend([feat[self.dialog_act_id] for caller, utt, feat in dialog if feat is not None])
        self.dialog_act_vocab = [t for t, cnt in Counter(all_dialog_acts).most_common()]
        self.rev_dialog_act_vocab = {t: idx for idx, t in enumerate(self.dialog_act_vocab)}

        all_personas = []
        for persona in self.train_corpus[self.persona_id]:
            for p in persona:
                all_personas.append(' '.join(p))

        self.persona_precursor_idx = []
        for i in self.persona_precursor:
            tmp = []
            for j in i.split(' '):
                if j != '':
                    tmp.append(self.rev_vocab[j])
            self.persona_precursor_idx.append(tmp)

    def load_word2vec(self):
        if self.word_vec_path is None:
            return
        with open(self.word_vec_path, "rb") as f:
            lines = f.readlines()
        raw_word2vec = {}
        for l in lines:
            w, vec = l.split(" ", 1)
            raw_word2vec[w] = vec
        self.word2vec = []
        oov_cnt = 0
        for v in self.vocab:
            str_vec = raw_word2vec.get(v, None)
            if str_vec is None:
                oov_cnt += 1
                vec = np.random.randn(self.word2vec_dim) * 0.1
            else:
                vec = np.fromstring(str_vec, sep=" ")
            self.word2vec.append(vec)
        print("word2vec cannot cover %f vocab" % (float(oov_cnt) / len(self.vocab)))

    def get_utt_corpus(self):
        def _to_id_corpus(data):
            results = []
            for line in data:
                results.append([self.rev_vocab.get(t, self.unk_id) for t in line])
            return results
        id_train = _to_id_corpus(self.train_corpus[self.utt_id])
        id_valid = _to_id_corpus(self.valid_corpus[self.utt_id])
        id_test = _to_id_corpus(self.test_corpus[self.utt_id])
        return {'train': id_train, 'valid': id_valid, 'test': id_test}

    def get_dialog_corpus(self):
        def _to_id_corpus(data):
            results = []
            for dialog in data:
                temp = []
                for utt, floor, feat in dialog:
                    if feat is not None:
                        id_feat = list(feat)
                        id_feat[self.dialog_act_id] = self.rev_dialog_act_vocab[feat[self.dialog_act_id]]
                    else:
                        id_feat = None
                    temp.append(([self.rev_vocab.get(t, self.unk_id) for t in utt], floor, id_feat))
                results.append(temp)
            return results

        id_train = _to_id_corpus(self.train_corpus[self.dialog_id])
        id_valid = _to_id_corpus(self.valid_corpus[self.dialog_id])
        id_test = _to_id_corpus(self.test_corpus[self.dialog_id])
        return {'train': id_train, 'valid': id_valid, 'test': id_test}

    def get_meta_corpus(self):
        def _to_id_corpus(data):
            results = []
            for m_meta, o_meta, topic in data:
                results.append((m_meta, o_meta, self.rev_topic_vocab[topic]))
            return results

        id_train = _to_id_corpus(self.train_corpus[self.meta_id])
        id_valid = _to_id_corpus(self.valid_corpus[self.meta_id])
        id_test = _to_id_corpus(self.test_corpus[self.meta_id])
        return {'train': id_train, 'valid': id_valid, 'test': id_test}

    def get_persona_corpus(self):
        def _to_id_corpus(data):
            results = []
            for i in data:
                session = []
                for j in i:
                    session.append([self.rev_vocab.get(k, self.unk_id) for k in j])
                results.append(session)
            return results

        id_train = _to_id_corpus(self.train_corpus[self.persona_id])
        id_valid = _to_id_corpus(self.valid_corpus[self.persona_id])
        id_test = _to_id_corpus(self.test_corpus[self.persona_id])
        return {'train': id_train, 'valid': id_valid, 'test': id_test}

    def get_persona_word_corpus(self):
        def _to_id_corpus(data):
            results = []
            for i in data:
                results.append([self.rev_vocab.get(k, self.unk_id) for k in i[0].strip().split(' ')])
            return results

        id_train = _to_id_corpus(self.train_corpus[self.persona_word_id])
        id_valid = _to_id_corpus(self.valid_corpus[self.persona_word_id])
        id_test = _to_id_corpus(self.test_corpus[self.persona_word_id])
        return {'train': id_train, 'valid': id_valid, 'test': id_test}
