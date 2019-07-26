import os
import time
import numpy as np
import tensorflow as tf
from config_api.config_utils import Config as Config
from data_apis.corpus import ConvAI2DialogCorpus
from data_apis.data_utils import ConvAI2DataLoader
from models.model import perCVAE
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-data", default="data/", help="ConvAI2 persona dialogue data directory.")
parser.add_argument("-vocab_file", default="convai2_vocab.txt", help="ConvAI2 persona dialogue vocabulary.")
parser.add_argument("-idf_file", default="convai2_voacb_idf.txt", help="ConvAI2 persona dialogue words' IDF.")
parser.add_argument("-embedding", default=None, help="The path to word2vec. Can be None.")
parser.add_argument("-save_to", default="saved_models", help="Experiment results directory.")
parser.add_argument("-train", action='store_true', help="Training model otherwise testing")
parser.add_argument("-test", action='store_true', help="Testing model")
parser.add_argument("-model", default=None, help="Trained model used in testing")
parser.add_argument("-config", default="without_labeled_data.yaml", help="Config for basic parameter setting")
args = parser.parse_args()

word2vec_path = args.embedding
data_dir = args.data
work_dir = args.save_to
test_path = args.model
vocab_file = args.vocab_file
idf_file = args.idf_file
para_config = args.config

forward_only = None
if args.train:
    forward_only = False
elif args.test:
    forward_only = True
if forward_only is None:
    print("Please specify training or testing by -train or -test")
    raise NameError

tf.app.flags.DEFINE_string("word2vec_path", word2vec_path, "The path to word2vec. Can be None.")
tf.app.flags.DEFINE_string("data_dir", data_dir, "ConvAI2 persona dialogue data directory.")
tf.app.flags.DEFINE_string("work_dir", work_dir, "Experiment results directory.")
tf.app.flags.DEFINE_string("test_path", test_path, "the dir to load checkpoint for forward only")
tf.app.flags.DEFINE_string("vocab_file", vocab_file, "the dir to load pre-processed vocabulary")
tf.app.flags.DEFINE_string("idf_file", idf_file, "the dir to load pre-processed words' IDF")
tf.app.flags.DEFINE_string("para_config", para_config, "the config name for para setting")
tf.app.flags.DEFINE_bool("forward_only", forward_only, "Only do decoding")
tf.app.flags.DEFINE_bool("equal_batch", True, "Make each batch has similar length.")
tf.app.flags.DEFINE_bool("resume", False, "Resume from previous")
tf.app.flags.DEFINE_bool("save_model", True, "Create checkpoints")
FLAGS = tf.app.flags.FLAGS


def main():
    config = Config(FLAGS.para_config)

    valid_config = Config(FLAGS.para_config)
    valid_config.keep_prob = 1.0
    valid_config.dec_keep_prob = 1.0
    valid_config.batch_size = 32

    test_config = Config(FLAGS.para_config)
    test_config.keep_prob = 1.0
    test_config.dec_keep_prob = 1.0
    test_config.batch_size = config.test_batchsize

    corpus = ConvAI2DialogCorpus(FLAGS.data_dir, max_vocab_cnt=config.vocab_size, word2vec=FLAGS.word2vec_path,
                                 word2vec_dim=config.embed_size, vocab_files=FLAGS.vocab_file, idf_files=FLAGS.idf_file)
    dial_corpus = corpus.get_dialog_corpus()
    meta_corpus = corpus.get_meta_corpus()
    persona_corpus = corpus.get_persona_corpus()
    persona_word_corpus = corpus.get_persona_word_corpus()
    vocab_size = corpus.gen_vocab_size
    vocab_idf = corpus.index2idf

    train_meta, valid_meta, test_meta = meta_corpus.get("train"), meta_corpus.get("valid"), meta_corpus.get("test")
    train_dial, valid_dial, test_dial = dial_corpus.get("train"), dial_corpus.get("valid"), dial_corpus.get("test")
    train_persona, valid_persona, test_persona = persona_corpus.get("train"), persona_corpus.get(
        "valid"), persona_corpus.get("test")
    train_persona_word, valid_persona_word, test_persona_word = persona_word_corpus.get(
        "train"), persona_word_corpus.get("valid"), persona_word_corpus.get("test")

    train_feed = ConvAI2DataLoader("Train", train_dial, train_meta, train_persona, train_persona_word, config,
                                   vocab_size, vocab_idf)
    valid_feed = ConvAI2DataLoader("Valid", valid_dial, valid_meta, valid_persona, valid_persona_word, config,
                                   vocab_size, vocab_idf)
    test_feed = ConvAI2DataLoader("Test", test_dial, test_meta, test_persona, test_persona_word, config, vocab_size,
                                  vocab_idf)

    if FLAGS.forward_only or FLAGS.resume:
        log_dir = os.path.join(FLAGS.test_path)
    else:
        log_dir = os.path.join(FLAGS.work_dir, "model" + time.strftime("_%Y_%m_%d_%H_%M_%S"))

    with tf.Session() as sess:
        initializer = tf.random_uniform_initializer(-1.0 * config.init_w, config.init_w)
        scope = "model"
        with tf.variable_scope(scope, reuse=None, initializer=initializer):
            model = perCVAE(sess, config, corpus, log_dir=None if FLAGS.forward_only else log_dir, forward=False,
                            scope=scope, name="Train")
        with tf.variable_scope(scope, reuse=True, initializer=initializer):
            valid_model = perCVAE(sess, valid_config, corpus, log_dir=None, forward=False, scope=scope, name="Valid")
        with tf.variable_scope(scope, reuse=True, initializer=initializer):
            test_model = perCVAE(sess, test_config, corpus, log_dir=None, forward=True, scope=scope, name="Test")

        print("Created computation graphs")
        if corpus.word2vec is not None and not FLAGS.forward_only:
            print("Loaded word2vec")
            sess.run(model.embedding.assign(np.array(corpus.word2vec)))
        ckp_dir = os.path.join(log_dir, "checkpoints")
        if not os.path.exists(ckp_dir):
            os.mkdir(ckp_dir)

        ckpt = tf.train.get_checkpoint_state(ckp_dir)
        if ckpt:
            print("Reading dm models parameters from %s" % ckpt.model_checkpoint_path)
            model.saver.restore(sess, ckpt.model_checkpoint_path)

        else:
            print("Created models with fresh parameters.")
            sess.run(tf.global_variables_initializer())

        if not FLAGS.forward_only:
            dm_checkpoint_path = os.path.join(ckp_dir, model.__class__.__name__ + ".ckpt")
            global_t = 1
            patience = 10
            dev_loss_threshold = np.inf
            best_dev_loss = np.inf
            for epoch in range(config.max_epoch):
                print(">> Epoch %d with lr %f" % (epoch, model.learning_rate.eval()))
                if train_feed.num_batch is None or train_feed.ptr >= train_feed.num_batch:
                    train_feed.epoch_init(config.batch_size, config.context_window,
                                          config.step_size, shuffle=True)
                global_t, train_loss = model.train(global_t, sess, train_feed, update_limit=config.update_limit)
                valid_feed.epoch_init(valid_config.batch_size, valid_config.context_window,
                                      valid_config.step_size, shuffle=False, intra_shuffle=False)
                valid_loss = valid_model.valid("ELBO_VALID", sess, valid_feed)

                test_feed.epoch_init(test_config.batch_size, test_config.context_window,
                                     test_config.step_size, shuffle=True, intra_shuffle=False)
                test_model.test(sess, test_feed, num_batch=5)
                done_epoch = epoch + 1

                if config.op == "sgd" and done_epoch > config.lr_hold:
                    sess.run(model.learning_rate_decay_op)

                if valid_loss < best_dev_loss:
                    if valid_loss <= dev_loss_threshold * config.improve_threshold:
                        patience = max(patience, done_epoch * config.patient_increase)
                        dev_loss_threshold = valid_loss
                        best_dev_loss = valid_loss

                if FLAGS.save_model:
                    print("Save model!!")
                    model.saver.save(sess, dm_checkpoint_path, global_step=epoch)

                if config.early_stop and patience <= done_epoch:
                    print("!!Early stop due to run out of patience!!")
                    break
            print("Best validation loss %f" % best_dev_loss)
            print("Done training")
        else:
            test_feed.epoch_init(test_config.batch_size, test_config.context_window,
                                 test_config.step_size, shuffle=False, intra_shuffle=False)
            test_model.test(sess, test_feed, num_batch=None, repeat=config.test_samples)


if __name__ == "__main__":
    if FLAGS.forward_only:
        if FLAGS.test_path is None:
            print("Set test_path before forward only")
            exit(1)
    main()
