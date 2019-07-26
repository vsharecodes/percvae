import os
import re
import sys
import time
import numpy as np
from tensorflow.contrib import layers
from tensorflow.contrib.rnn import OutputProjectionWrapper
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import variable_scope
import rnn_cell_impl as rnn_cell
from models.dynamic_rnn_decoder import dynamic_rnn_decoder
from utils import gaussian_kld
from utils import get_bi_rnn_encode
from utils import get_rnn_encode
from utils import get_bow
from utils import norm_log_liklihood
from utils import sample_gaussian
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import nest
import tensorflow as tf


class BaseTFModel(object):
    global_t = tf.placeholder(dtype=tf.int32, name="global_t")
    learning_rate = None
    scope = None

    @staticmethod
    def print_model_stats(tvars):
        total_parameters = 0
        for variable in tvars:
            shape = variable.get_shape()
            variable_parametes = 1
            for dim in shape:
                variable_parametes *= dim.value
            print("Trainable %s with %d parameters" % (variable.name, variable_parametes))
            total_parameters += variable_parametes
        print("Total number of trainable parameters is %d" % total_parameters)

    @staticmethod
    def get_rnncell(cell_type, cell_size, keep_prob, num_layer):
        cells = []
        for _ in range(num_layer):
            if cell_type == "gru":
                cell = rnn_cell.GRUCell(cell_size)
            else:
                cell = rnn_cell.LSTMCell(cell_size, use_peepholes=False, forget_bias=1.0)

            if keep_prob < 1.0:
                cell = rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)

            cells.append(cell)

        if num_layer > 1:
            cell = rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
        else:
            cell = cells[0]

        return cell

    @staticmethod
    def print_loss(prefix, loss_names, losses, postfix):
        template = "%s "
        for name in loss_names:
            template += "%s " % name
            template += " %f "
        template += "%s"
        template = re.sub(' +', ' ', template)
        avg_losses = []
        values = [prefix]

        for loss in losses:
            values.append(np.mean(loss))
            avg_losses.append(np.mean(loss))
        values.append(postfix)

        print(template % tuple(values))
        return avg_losses

    def train(self, global_t, sess, train_feed):
        raise NotImplementedError("Train function needs to be implemented")

    def valid(self, *args, **kwargs):
        raise NotImplementedError("Valid function needs to be implemented")

    def batch_2_feed(self, *args, **kwargs):
        raise NotImplementedError("Implement how to unpack the back")

    def optimize(self, sess, config, loss, log_dir):
        if log_dir is None:
            return
        # optimization
        if self.scope is None:
            tvars = tf.trainable_variables()
        else:
            tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)
        grads = tf.gradients(loss, tvars)
        if config.grad_clip is not None:
            grads, _ = tf.clip_by_global_norm(grads, tf.constant(config.grad_clip))
        # add gradient noise
        if config.grad_noise > 0:
            grad_std = tf.sqrt(config.grad_noise / tf.pow(1.0 + tf.to_float(self.global_t), 0.55))
            grads = [g + tf.truncated_normal(tf.shape(g), mean=0.0, stddev=grad_std) for g in grads]

        if config.op == "adam":
            print("Use Adam")
            optimizer = tf.train.AdamOptimizer(config.init_lr)
        elif config.op == "rmsprop":
            print("Use RMSProp")
            optimizer = tf.train.RMSPropOptimizer(config.init_lr)
        else:
            print("Use SGD")
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)

        self.train_ops = optimizer.apply_gradients(zip(grads, tvars))
        self.print_model_stats(tvars)
        train_log_dir = os.path.join(log_dir, "checkpoints")
        print("Save summary to %s" % log_dir)
        self.train_summary_writer = tf.summary.FileWriter(train_log_dir, sess.graph)


class perCVAE(BaseTFModel):

    def __init__(self, sess, config, api, log_dir, forward, scope=None, name=None):
        self.vocab = api.vocab
        self.rev_vocab = api.rev_vocab
        self.vocab_size = len(self.vocab)
        self.idf = api.index2idf
        self.gen_vocab_size = api.gen_vocab_size
        self.topic_vocab = api.topic_vocab
        self.topic_vocab_size = len(self.topic_vocab)
        self.da_vocab = api.dialog_act_vocab
        self.da_vocab_size = len(self.da_vocab)
        self.sess = sess
        self.scope = scope
        self.max_utt_len = config.max_utt_len
        self.max_per_len = config.max_per_len
        self.max_per_line = config.max_per_line
        self.max_per_words = config.max_per_words
        self.go_id = self.rev_vocab["<s>"]
        self.eos_id = self.rev_vocab["</s>"]
        self.context_cell_size = config.cxt_cell_size
        self.sent_cell_size = config.sent_cell_size
        self.memory_cell_size = config.memory_cell_size
        self.dec_cell_size = config.dec_cell_size
        self.hops = config.hops
        self.batch_size = config.batch_size
        self.test_samples = config.test_samples
        self.balance_factor = config.balance_factor

        with tf.name_scope("io"):
            self.first_dimension_size = self.batch_size
            self.input_contexts = tf.placeholder(dtype=tf.int32,
                                                 shape=(self.first_dimension_size, None, self.max_utt_len),
                                                 name="dialog_context")
            self.floors = tf.placeholder(dtype=tf.int32, shape=(self.first_dimension_size, None), name="floor")
            self.context_lens = tf.placeholder(dtype=tf.int32, shape=(self.first_dimension_size,), name="context_lens")
            self.topics = tf.placeholder(dtype=tf.int32, shape=(self.first_dimension_size,), name="topics")
            self.personas = tf.placeholder(dtype=tf.int32,
                                           shape=(self.first_dimension_size, self.max_per_line, self.max_per_len),
                                           name="personas")
            self.persona_words = tf.placeholder(dtype=tf.int32,
                                                shape=(self.first_dimension_size, self.max_per_line, self.max_per_len),
                                                name="persona_words")
            self.persona_position = tf.placeholder(dtype=tf.int32, shape=(self.first_dimension_size, None),
                                                   name="persona_position")
            self.selected_persona = tf.placeholder(dtype=tf.int32, shape=(self.first_dimension_size, 1),
                                                   name="selected_persona")

            self.query = tf.placeholder(dtype=tf.int32, shape=(self.first_dimension_size, self.max_utt_len),
                                        name="query")

            # target response given the dialog context
            self.output_tokens = tf.placeholder(dtype=tf.int32, shape=(self.first_dimension_size, None),
                                                name="output_token")
            self.output_lens = tf.placeholder(dtype=tf.int32, shape=(self.first_dimension_size,), name="output_lens")

            # optimization related variables
            self.learning_rate = tf.Variable(float(config.init_lr), trainable=False, name="learning_rate")
            self.learning_rate_decay_op = self.learning_rate.assign(tf.multiply(self.learning_rate, config.lr_decay))
            self.global_t = tf.placeholder(dtype=tf.int32, name="global_t")
            self.use_prior = tf.placeholder(dtype=tf.bool, name="use_prior")

        max_context_lines = array_ops.shape(self.input_contexts)[1]
        max_out_len = array_ops.shape(self.output_tokens)[1]
        batch_size = array_ops.shape(self.input_contexts)[0]

        with variable_scope.variable_scope("wordEmbedding"):
            self.embedding = tf.get_variable("embedding", [self.vocab_size, config.embed_size], dtype=tf.float32)
            embedding_mask = tf.constant([0 if i == 0 else 1 for i in range(self.vocab_size)], dtype=tf.float32,
                                         shape=[self.vocab_size, 1])
            embedding = self.embedding * embedding_mask
            input_embedding = embedding_ops.embedding_lookup(embedding, tf.reshape(self.input_contexts, [-1]))
            input_embedding = tf.reshape(input_embedding, [-1, self.max_utt_len, config.embed_size])
            output_embedding = embedding_ops.embedding_lookup(embedding, self.output_tokens)
            persona_input_embedding = embedding_ops.embedding_lookup(embedding, tf.reshape(self.personas, [-1]))
            persona_input_embedding = tf.reshape(persona_input_embedding,
                                                 [-1, self.max_per_len, config.embed_size])
            if config.sent_type == "bow":
                input_embedding, sent_size = get_bow(input_embedding)
                output_embedding, _ = get_bow(output_embedding)
                persona_input_embedding, _ = get_bow(persona_input_embedding)

            elif config.sent_type == "rnn":
                sent_cell = self.get_rnncell("gru", self.sent_cell_size, config.keep_prob, 1)
                _, input_embedding, sent_size = get_rnn_encode(input_embedding, sent_cell, scope="sent_rnn")
                _, output_embedding, _ = get_rnn_encode(output_embedding, sent_cell, self.output_lens, scope="sent_rnn",
                                                        reuse=True)
                _, persona_input_embedding, _ = get_rnn_encode(persona_input_embedding, sent_cell, scope="sent_rnn",
                                                               reuse=True)
            elif config.sent_type == "bi_rnn":
                fwd_sent_cell = self.get_rnncell("gru", self.sent_cell_size, keep_prob=1.0, num_layer=1)
                bwd_sent_cell = self.get_rnncell("gru", self.sent_cell_size, keep_prob=1.0, num_layer=1)
                input_step_embedding, input_embedding, sent_size = get_bi_rnn_encode(input_embedding, fwd_sent_cell,
                                                                                     bwd_sent_cell, scope="sent_bi_rnn")
                _, output_embedding, _ = get_bi_rnn_encode(output_embedding, fwd_sent_cell, bwd_sent_cell,
                                                           self.output_lens, scope="sent_bi_rnn", reuse=True)
                _, persona_input_embedding, _ = get_bi_rnn_encode(persona_input_embedding, fwd_sent_cell, bwd_sent_cell,
                                                                  scope="sent_bi_rnn", reuse=True)
            else:
                raise ValueError("Unknown sent_type. Must be one of [bow, rnn, bi_rnn]")
            # reshape input into dialogs
            input_embedding = tf.reshape(input_embedding, [-1, max_context_lines, sent_size])
            self.input_step_embedding = input_step_embedding
            self.encoder_state_size = sent_size
            if config.keep_prob < 1.0:
                input_embedding = tf.nn.dropout(input_embedding, config.keep_prob)

        with variable_scope.variable_scope("personaMemory"):
            embedding_mask = tf.constant([0 if i == 0 else 1 for i in range(self.vocab_size)], dtype=tf.float32,
                                         shape=[self.vocab_size, 1])
            A = tf.get_variable("persona_embedding_A", [self.vocab_size, self.memory_cell_size], dtype=tf.float32)
            A = A * embedding_mask
            C = []
            for hopn in range(self.hops):
                C.append(
                    tf.get_variable("persona_embedding_C_hop_{}".format(hopn), [self.vocab_size, self.memory_cell_size],
                                    dtype=tf.float32) * embedding_mask)

            q_emb = tf.nn.embedding_lookup(A, self.query)
            u_0 = tf.reduce_sum(q_emb, 1)
            u = [u_0]
            for hopn in range(self.hops):
                if hopn == 0:
                    m_emb_A = tf.nn.embedding_lookup(A, self.personas)
                    m_A = tf.reshape(m_emb_A, [-1, self.max_per_len * self.max_per_line, self.memory_cell_size])
                else:
                    with tf.variable_scope('persona_hop_{}'.format(hopn)):
                        m_emb_A = tf.nn.embedding_lookup(C[hopn - 1], self.personas)
                        m_A = tf.reshape(m_emb_A, [-1, self.max_per_len * self.max_per_line, self.memory_cell_size])
                u_temp = tf.transpose(tf.expand_dims(u[-1], -1), [0, 2, 1])
                dotted = tf.reduce_sum(m_A * u_temp, 2)
                probs = tf.nn.softmax(dotted)
                probs_temp = tf.transpose(tf.expand_dims(probs, -1), [0, 2, 1])
                with tf.variable_scope('persona_hop_{}'.format(hopn)):
                    m_emb_C = tf.nn.embedding_lookup(C[hopn], tf.reshape(self.personas,
                                                                         [-1, self.max_per_len * self.max_per_line]))
                    m_emb_C = tf.expand_dims(m_emb_C, -2)
                    m_C = tf.reduce_sum(m_emb_C, axis=2)
                c_temp = tf.transpose(m_C, [0, 2, 1])
                o_k = tf.reduce_sum(c_temp * probs_temp, axis=2)
                u_k = u[-1] + o_k
                u.append(u_k)
            persona_memory = u[-1]

        with variable_scope.variable_scope("contextEmbedding"):
            context_layers = 2
            enc_cell = self.get_rnncell(config.cell_type, self.context_cell_size, keep_prob=1.0,
                                        num_layer=context_layers)
            _, enc_last_state = tf.nn.dynamic_rnn(
                enc_cell,
                input_embedding,
                dtype=tf.float32,
                sequence_length=self.context_lens)

            if context_layers > 1:
                if config.cell_type == 'lstm':
                    enc_last_state = [temp.h for temp in enc_last_state]

                enc_last_state = tf.concat(enc_last_state, 1)
            else:
                if config.cell_type == 'lstm':
                    enc_last_state = enc_last_state.h

        cond_embedding = tf.concat([persona_memory, enc_last_state], 1)

        with variable_scope.variable_scope("recognitionNetwork"):
            recog_input = tf.concat([cond_embedding, output_embedding, persona_memory], 1)
            self.recog_mulogvar = recog_mulogvar = layers.fully_connected(recog_input, config.latent_size * 2,
                                                                          activation_fn=None, scope="muvar")
            recog_mu, recog_logvar = tf.split(recog_mulogvar, 2, axis=1)

        with variable_scope.variable_scope("priorNetwork"):
            prior_fc1 = layers.fully_connected(cond_embedding, np.maximum(config.latent_size * 2, 100),
                                               activation_fn=tf.tanh, scope="fc1")
            prior_mulogvar = layers.fully_connected(prior_fc1, config.latent_size * 2, activation_fn=None,
                                                    scope="muvar")
            prior_mu, prior_logvar = tf.split(prior_mulogvar, 2, axis=1)
            latent_sample = tf.cond(self.use_prior,
                                    lambda: sample_gaussian(prior_mu, prior_logvar),
                                    lambda: sample_gaussian(recog_mu, recog_logvar))

        with variable_scope.variable_scope("personaSelecting"):
            condition = tf.concat([persona_memory, latent_sample], 1)

            self.persona_dist = tf.nn.log_softmax(
                layers.fully_connected(condition, self.max_per_line, activation_fn=tf.tanh, scope="persona_dist"))
            select_temp = tf.expand_dims(tf.argmax(self.persona_dist, 1, output_type=tf.int32), 1)
            index_temp = tf.expand_dims(tf.range(0, self.first_dimension_size, dtype=tf.int32), 1)
            persona_select = tf.concat([index_temp, select_temp], 1)
            selected_words_ordered = tf.reshape(tf.gather_nd(self.persona_words, persona_select),
                                                [self.max_per_len * self.first_dimension_size])
            self.selected_words = tf.gather_nd(self.persona_words, persona_select)
            label = tf.reshape(selected_words_ordered, [self.max_per_len * self.first_dimension_size, 1])
            index = tf.reshape(tf.range(self.first_dimension_size, dtype=tf.int32), [self.first_dimension_size, 1])
            index = tf.reshape(tf.tile(index, [1, self.max_per_len]), [self.max_per_len * self.first_dimension_size, 1])

            concated = tf.concat([index, label], 1)
            true_labels = tf.where(selected_words_ordered > 0)
            concated = tf.gather_nd(concated, true_labels)
            self.persona_word_mask = tf.sparse_to_dense(concated, [self.first_dimension_size, self.vocab_size],
                                                        config.perw_weight, 0.0)
            self.other_word_mask = tf.sparse_to_dense(concated, [self.first_dimension_size, self.vocab_size], 0.0,
                                                      config.othw_weight)
            self.persona_word_mask = self.persona_word_mask * self.idf

        with variable_scope.variable_scope("generationNetwork"):
            gen_inputs = tf.concat([cond_embedding, latent_sample], 1)

            # BOW loss
            bow_fc1 = layers.fully_connected(gen_inputs, 400, activation_fn=tf.tanh, scope="bow_fc1")
            if config.keep_prob < 1.0:
                bow_fc1 = tf.nn.dropout(bow_fc1, config.keep_prob)
            self.bow_logits = layers.fully_connected(bow_fc1, self.vocab_size, activation_fn=None, scope="bow_project")

            # Y loss
            dec_inputs = gen_inputs
            selected_attribute_embedding = None
            self.da_logits = tf.zeros((batch_size, self.da_vocab_size))

            # Decoder
            if config.num_layer > 1:
                dec_init_state = []
                for i in range(config.num_layer):
                    temp_init = layers.fully_connected(dec_inputs, self.dec_cell_size, activation_fn=None,
                                                       scope="init_state-%d" % i)
                    if config.cell_type == 'lstm':
                        temp_init = rnn_cell.LSTMStateTuple(temp_init, temp_init)

                    dec_init_state.append(temp_init)

                dec_init_state = tuple(dec_init_state)
            else:
                dec_init_state = layers.fully_connected(dec_inputs, self.dec_cell_size, activation_fn=None,
                                                        scope="init_state")
                if config.cell_type == 'lstm':
                    dec_init_state = rnn_cell.LSTMStateTuple(dec_init_state, dec_init_state)

        with variable_scope.variable_scope("decoder"):
            dec_cell = self.get_rnncell(config.cell_type, self.dec_cell_size, config.keep_prob, config.num_layer)
            dec_cell = OutputProjectionWrapper(dec_cell, self.vocab_size)

            pos_cell = self.get_rnncell(config.cell_type, self.dec_cell_size, config.keep_prob, config.num_layer)
            pos_cell = OutputProjectionWrapper(pos_cell, self.vocab_size)

            with variable_scope.variable_scope("position"):
                self.pos_w_1 = tf.get_variable("pos_w_1",
                                               [self.dec_cell_size, 2],
                                               dtype=tf.float32)
                self.pos_b_1 = tf.get_variable("pos_b_1",
                                               [2],
                                               dtype=tf.float32)

            def position_function(states, logp=False):
                states = tf.reshape(states, [-1, self.dec_cell_size])
                if logp:
                    return tf.reshape(
                        tf.nn.log_softmax(tf.matmul(states, self.pos_w_1) + self.pos_b_1),
                        [self.first_dimension_size, -1, 2])
                return tf.reshape(
                        tf.nn.softmax(tf.matmul(states, self.pos_w_1) + self.pos_b_1),
                        [self.first_dimension_size, -1, 2])

            if forward:
                loop_func = self.context_decoder_fn_inference(position_function,
                                                              self.persona_word_mask,
                                                              self.other_word_mask,
                                                              None, dec_init_state, embedding,
                                                              start_of_sequence_id=self.go_id,
                                                              end_of_sequence_id=self.eos_id,
                                                              maximum_length=self.max_utt_len,
                                                              num_decoder_symbols=self.vocab_size,
                                                              context_vector=selected_attribute_embedding,
                                                              )
                dec_input_embedding = None
                dec_seq_lens = None
            else:
                loop_func = self.context_decoder_fn_train(dec_init_state,
                                                          selected_attribute_embedding)
                dec_input_embedding = embedding_ops.embedding_lookup(embedding, self.output_tokens)
                dec_input_embedding = dec_input_embedding[:, 0:-1, :]
                dec_seq_lens = self.output_lens - 1
                if config.keep_prob < 1.0:
                    dec_input_embedding = tf.nn.dropout(dec_input_embedding, config.keep_prob)
                if config.dec_keep_prob < 1.0:
                    keep_mask = tf.less_equal(tf.random_uniform((batch_size, max_out_len - 1), minval=0.0, maxval=1.0),
                                              config.dec_keep_prob)
                    keep_mask = tf.expand_dims(tf.to_float(keep_mask), 2)
                    dec_input_embedding = dec_input_embedding * keep_mask
                    dec_input_embedding = tf.reshape(dec_input_embedding, [-1, max_out_len - 1, config.embed_size])

            with variable_scope.variable_scope("dec_state"):
                dec_outs, _, final_context_state, rnn_states = dynamic_rnn_decoder(dec_cell, loop_func,
                                                                                   inputs=dec_input_embedding,
                                                                                   sequence_length=dec_seq_lens)
            with variable_scope.variable_scope("pos_state"):
                _, _, _, pos_states = dynamic_rnn_decoder(pos_cell, loop_func, inputs=dec_input_embedding,
                                                          sequence_length=dec_seq_lens)

            self.position_dist = position_function(pos_states, logp=True)

            if final_context_state is not None:
                final_context_state = final_context_state[:, 0:array_ops.shape(dec_outs)[1]]
                mask = tf.to_int32(tf.sign(tf.reduce_max(dec_outs, axis=2)))
                self.dec_out_words = tf.multiply(tf.reverse(final_context_state, axis=[1]), mask)
            else:
                self.dec_out_words = tf.argmax(dec_outs, 2)
        if not forward:
            with variable_scope.variable_scope("loss"):
                labels = self.output_tokens[:, 1:]
                label_mask = tf.to_float(tf.sign(labels))
                rc_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=dec_outs, labels=labels)
                rc_loss = tf.reduce_sum(rc_loss * label_mask, reduction_indices=1)
                self.avg_rc_loss = tf.reduce_mean(rc_loss)
                self.rc_ppl = tf.exp(tf.reduce_sum(rc_loss) / tf.reduce_sum(label_mask))
                per_select_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=tf.reshape(self.persona_dist, [self.first_dimension_size, 1, -1]),
                    labels=self.selected_persona)
                per_select_loss = tf.reduce_sum(per_select_loss, reduction_indices=1)
                self.avg_per_select_loss = tf.reduce_mean(per_select_loss)
                position_labels = self.persona_position[:, 1:]
                per_pos_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.position_dist,
                                                                              labels=position_labels)
                per_pos_loss = tf.reduce_sum(per_pos_loss, reduction_indices=1)
                self.avg_per_pos_loss = tf.reduce_mean(per_pos_loss)

                tile_bow_logits = tf.tile(tf.expand_dims(self.bow_logits, 1), [1, max_out_len - 1, 1])
                bow_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tile_bow_logits,
                                                                          labels=labels) * label_mask
                bow_loss = tf.reduce_sum(bow_loss, reduction_indices=1)
                self.avg_bow_loss = tf.reduce_mean(bow_loss)
                kld = gaussian_kld(recog_mu, recog_logvar, prior_mu, prior_logvar)
                self.avg_kld = tf.reduce_mean(kld)

                if log_dir is not None:
                    kl_weights = tf.minimum(tf.to_float(self.global_t) / config.full_kl_step, 1.0)
                else:
                    kl_weights = tf.constant(1.0)

                self.kl_w = kl_weights
                self.elbo = self.avg_rc_loss + kl_weights * self.avg_kld
                aug_elbo = self.elbo + self.avg_bow_loss + 0.1 * self.avg_per_select_loss + 0.05 * self.avg_per_pos_loss

                tf.summary.scalar("rc_loss", self.avg_rc_loss)
                tf.summary.scalar("elbo", self.elbo)
                tf.summary.scalar("kld", self.avg_kld)
                tf.summary.scalar("per_pos_loss", self.avg_per_pos_loss)

                self.summary_op = tf.summary.merge_all()

                self.log_p_z = norm_log_liklihood(latent_sample, prior_mu, prior_logvar)
                self.log_q_z_xy = norm_log_liklihood(latent_sample, recog_mu, recog_logvar)
                self.est_marginal = tf.reduce_mean(rc_loss + bow_loss - self.log_p_z + self.log_q_z_xy)

            self.optimize(sess, config, aug_elbo, log_dir)

        self.saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2)

    def position_encoding(self, sentence_size, embedding_size):
        encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
        ls = sentence_size + 1
        le = embedding_size + 1
        for i in range(1, le):
            for j in range(1, ls):
                encoding[i - 1, j - 1] = (i - (embedding_size + 1) / 2) * (j - (sentence_size + 1) / 2)
        encoding = 1 + 4 * encoding / embedding_size / sentence_size
        encoding[:, -1] = 1.0
        return np.transpose(encoding)

    def batch_2_feed(self, batch, global_t, use_prior, repeat=1):
        context, context_lens, floors, topics, _, _, outputs, output_lens, _, personas, persona_words, persona_position, selected_persona = batch
        c_shape = context.shape
        query = np.zeros((c_shape[0], c_shape[2]), dtype=int)
        for i in range(c_shape[0]):
            query[i] = context[i][-1]
        feed_dict = {self.input_contexts: context, self.context_lens: context_lens,
                     self.floors: floors, self.topics: topics,
                     self.output_tokens: outputs,
                     self.output_lens: output_lens,
                     self.use_prior: use_prior,
                     self.personas: personas,
                     self.persona_words: persona_words,
                     self.query: query,
                     self.persona_position: persona_position,
                     self.selected_persona: selected_persona}
        if repeat > 1:
            tiled_feed_dict = {}
            for key, val in feed_dict.items():
                if key is self.use_prior:
                    tiled_feed_dict[key] = val
                    continue
                multipliers = [1] * len(val.shape)
                multipliers[0] = repeat
                tiled_feed_dict[key] = np.tile(val, multipliers)
            feed_dict = tiled_feed_dict

        if global_t is not None:
            feed_dict[self.global_t] = global_t

        return feed_dict

    def train(self, global_t, sess, train_feed, update_limit=5000):
        elbo_losses = []
        rc_losses = []
        kl_losses = []
        per_pos_losses = []

        local_t = 0
        start_time = time.time()
        loss_names = ["elbo_loss", "rc_loss", "kl_loss", "per_pos_loss"]
        while True:
            batch = train_feed.next_batch()
            if batch is None:
                break
            if update_limit is not None and local_t >= update_limit:
                break
            feed_dict = self.batch_2_feed(batch, global_t, use_prior=False)

            _, sum_op, elbo_loss, rc_loss, kl_loss, per_pos_loss = sess.run(
                [self.train_ops, self.summary_op, self.elbo, self.avg_rc_loss,
                 self.avg_kld, self.avg_per_pos_loss], feed_dict)
            self.train_summary_writer.add_summary(sum_op, global_t)
            elbo_losses.append(elbo_loss)
            rc_losses.append(rc_loss)
            kl_losses.append(kl_loss)
            per_pos_losses.append(per_pos_loss)

            global_t += 1
            local_t += 1
            if local_t % 10 == 1:
                kl_w = sess.run(self.kl_w, {self.global_t: global_t})
                self.print_loss("%.2f" % (train_feed.ptr / float(train_feed.num_batch)), loss_names,
                                [elbo_losses, rc_losses, kl_losses, per_pos_losses],
                                "kl_weight %f" % kl_w)

        epoch_time = time.time() - start_time
        avg_losses = self.print_loss("Epoch Done", loss_names,
                                     [elbo_losses, rc_losses, kl_losses, per_pos_losses],
                                     "step time %.4f" % (epoch_time / train_feed.num_batch))

        return global_t, avg_losses[0]

    def valid(self, name, sess, valid_feed):
        elbo_losses = []
        rc_losses = []
        kl_losses = []
        per_pos_losses = []

        while True:
            batch = valid_feed.next_batch()
            if batch is None:
                break
            feed_dict = self.batch_2_feed(batch, None, use_prior=False, repeat=1)

            elbo_loss, rc_loss, kl_loss, per_pos_loss = sess.run(
                [self.elbo, self.avg_rc_loss,
                 self.avg_kld, self.avg_per_pos_loss], feed_dict)
            elbo_losses.append(elbo_loss)
            rc_losses.append(rc_loss)
            kl_losses.append(kl_loss)
            per_pos_losses.append(per_pos_loss)

        avg_losses = self.print_loss(name,
                                     ["elbo_loss", "rc_loss", "kl_loss", "per_pos_loss"],
                                     [elbo_losses, rc_losses, kl_losses, per_pos_losses], "")
        return avg_losses[0]

    def test(self, sess, test_feed, num_batch=None, repeat=5, dest=sys.stdout):
        local_t = 0

        while True:
            batch = test_feed.next_batch()
            if batch is None or (num_batch is not None and local_t > num_batch):
                break
            feed_dict = self.batch_2_feed(batch, None, use_prior=True, repeat=1)
            sample_words = []
            for i in range(repeat):
                word_outs = sess.run([self.dec_out_words], feed_dict)[0]
                sample_words.append(word_outs)

            true_srcs = feed_dict[self.input_contexts]
            true_src_lens = feed_dict[self.context_lens]
            true_outs = feed_dict[self.output_tokens]
            personas = feed_dict[self.personas]
            local_t += 1

            if dest != sys.stdout:
                if local_t % (test_feed.num_batch / 10) == 0:
                    print("%.2f >> " % (test_feed.ptr / float(test_feed.num_batch))),

            for b_id in range(test_feed.batch_size):
                # print the personas
                for t_id in range(0, self.max_per_line, 1):
                    persona = " ".join([self.vocab[e] for e in personas[b_id, t_id].tolist() if e != 0])
                    if persona.strip() is not '':
                        dest.write("Persona %d: %s\n" % (t_id, persona))

                # print the dialog context
                dest.write("Batch %d index %d\n" % (local_t, b_id))
                start = np.maximum(0, true_src_lens[b_id] - 5)
                for t_id in range(start, true_srcs.shape[1], 1):
                    src_str = " ".join([self.vocab[e] for e in true_srcs[b_id, t_id].tolist() if e != 0])
                    dest.write("Source: %s\n" % src_str)
                true_tokens = [self.vocab[e] for e in true_outs[b_id].tolist() if e not in [0, self.eos_id, self.go_id]]
                true_str = " ".join(true_tokens).replace(" ' ", "'")
                dest.write("Target >> %s\n" % (true_str))
                local_tokens = []
                for r_id in range(repeat):
                    pred_outs = sample_words[r_id]
                    pred_tokens = [self.vocab[e] for e in pred_outs[b_id].tolist() if e != self.eos_id and e != 0]
                    pred_str = " ".join(pred_tokens).replace(" ' ", "'")
                    dest.write("Sample %d >> %s\n" % (r_id, pred_str))
                    local_tokens.append(pred_tokens)
                print('\n')

        print("Done testing")

    def eval(self, name, sess, eval_feed):
        rc_losses = []
        rc_ppls = []
        while True:
            batch = eval_feed.next_batch()
            if batch is None:
                break
            feed_dict = self.batch_2_feed(batch, None, use_prior=True, repeat=1)
            rc_loss, rc_ppl = sess.run(
                [self.avg_rc_loss,
                 self.rc_ppl], feed_dict)
            rc_losses.append(rc_loss)
            rc_ppls.append(rc_ppl)
        avg_losses = self.print_loss(name, ["rc_loss", "rc_peplexity"],
                                     [rc_losses, rc_ppls], "")
        return avg_losses[0]

    def context_decoder_fn_train(self, encoder_state, context_vector, name=None):
        with ops.name_scope(name, "simple_decoder_fn_train", [encoder_state]):
            pass

        def decoder_fn(time, cell_state, cell_input, cell_output, context_state):
            with ops.name_scope(name, "simple_decoder_fn_train",
                                [time, cell_state, cell_input, cell_output,
                                 context_state]):

                cell_state = cell_state if cell_state is not None else encoder_state

                if context_vector is not None:
                    cell_input = tf.concat([cell_input, context_vector], axis=1)

                if cell_state is None:
                    return None, encoder_state, cell_input, cell_output, context_state
                else:
                    return None, cell_state, cell_input, cell_output, context_state

        return decoder_fn

    def context_decoder_fn_inference(self, position_fn, persona_word_mask, other_word_mask, output_fn,
                                     encoder_state, embeddings, start_of_sequence_id, end_of_sequence_id,
                                     maximum_length, num_decoder_symbols, context_vector,
                                     dtype=dtypes.int32, name=None, decode_type='greedy'):
        with ops.name_scope(name, "simple_decoder_fn_inference",
                            [position_fn, persona_word_mask,
                             other_word_mask, output_fn, encoder_state, embeddings,
                             start_of_sequence_id, end_of_sequence_id,
                             maximum_length, num_decoder_symbols, dtype]):
            start_of_sequence_id = ops.convert_to_tensor(start_of_sequence_id, dtype)
            end_of_sequence_id = ops.convert_to_tensor(end_of_sequence_id, dtype)
            maxium_length_int = maximum_length + 1
            maximum_length = ops.convert_to_tensor(maximum_length, dtype)
            num_decoder_symbols = ops.convert_to_tensor(num_decoder_symbols, dtype)
            encoder_info = nest.flatten(encoder_state)[0]
            batch_size = encoder_info.get_shape()[0].value
            if output_fn is None:
                output_fn = lambda x: x
            if batch_size is None:
                batch_size = array_ops.shape(encoder_info)[0]

        def decoder_fn(time, cell_state, cell_input, cell_output, context_state):
            with ops.name_scope(name, "simple_decoder_fn_inference",
                                [time, cell_state, cell_input, cell_output,
                                 context_state]):
                if cell_input is not None:
                    raise ValueError("Expected cell_input to be None, but saw: %s" %
                                     cell_input)
                if cell_output is None:
                    next_input_id = array_ops.ones([batch_size, ], dtype=dtype) * (
                        start_of_sequence_id)
                    done = array_ops.zeros([batch_size, ], dtype=dtypes.bool)
                    cell_state = encoder_state
                    cell_output = array_ops.zeros([num_decoder_symbols],
                                                  dtype=dtypes.float32)
                    context_state = tf.zeros((batch_size, maxium_length_int), dtype=tf.int32)
                else:
                    cell_output = output_fn(cell_output)

                    v = tf.squeeze(tf.log(position_fn(tf.expand_dims(cell_state, 1))), 1)
                    t = tf.exp(self.balance_factor)
                    delta = (tf.exp(v / t)) / (tf.reduce_sum(tf.exp(v / t), 1, keep_dims=True))
                    other_prob, per_prob = tf.split(delta, [1, 1], 1)
                    other_word_prob = other_word_mask * other_prob
                    persona_word_prob = persona_word_mask * per_prob
                    cell_output = cell_output * (other_word_prob + persona_word_prob)

                    if decode_type == 'sample':
                        matrix_U = -1.0 * tf.log(
                            -1.0 * tf.log(tf.random_uniform(tf.shape(cell_output), minval=0.0, maxval=1.0)))
                        next_input_id = math_ops.cast(
                            tf.argmax(tf.subtract(cell_output, matrix_U), dimension=1), dtype=dtype)
                    elif decode_type == 'greedy':
                        next_input_id = math_ops.cast(
                            math_ops.argmax(cell_output, 1), dtype=dtype)
                    else:
                        raise ValueError("unknown decode type")

                    done = math_ops.equal(next_input_id, end_of_sequence_id)
                    expanded_next_input = tf.expand_dims(next_input_id, axis=1)
                    sliced_context_state = tf.slice(context_state, [0, 0], [-1, maxium_length_int - 1])
                    context_state = tf.concat([expanded_next_input, sliced_context_state], axis=1)
                    context_state = tf.reshape(context_state, [batch_size, maxium_length_int])

                next_input = array_ops.gather(embeddings, next_input_id)
                if context_vector is not None:
                    next_input = tf.concat([next_input, context_vector], axis=1)
                done = control_flow_ops.cond(math_ops.greater(time, maximum_length),
                                             lambda: array_ops.ones([batch_size, ], dtype=dtypes.bool),
                                             lambda: done)
                return done, cell_state, next_input, cell_output, context_state

        return decoder_fn
