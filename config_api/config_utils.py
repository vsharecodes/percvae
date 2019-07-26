import yaml
import os


class Config(object):

    def __init__(self, para_config):
        curPath = os.path.dirname(os.path.realpath(__file__))
        yamlPath = os.path.join(curPath, para_config)
        self.cfg_dic = None
        with open(yamlPath, 'r') as config:
            cfg = config.read()
            self.cfg_dic = yaml.safe_load(cfg)
        if not self.cfg_dic:
            raise ValueError("Invalid config yaml name or yaml file doesn't exist.")
        self.update_limit = self.cfg_dic['update_limit']
        self.vocab_size = self.cfg_dic['vocab_size']
        self.sent_type = self.cfg_dic['sent_type']
        self.latent_size = self.cfg_dic['latent_size']
        self.full_kl_step = self.cfg_dic['full_kl_step']
        self.dec_keep_prob = self.cfg_dic['dec_keep_prob']
        self.cell_type = self.cfg_dic['cell_type']
        self.embed_size = self.cfg_dic['embed_size']
        self.cxt_cell_size = self.cfg_dic['cxt_cell_size']
        self.sent_cell_size = self.cfg_dic['sent_cell_size']
        self.memory_cell_size = self.cfg_dic['memory_cell_size']
        self.dec_cell_size = self.cfg_dic['dec_cell_size']
        self.context_window = self.cfg_dic['context_window']
        self.step_size = self.cfg_dic['step_size']
        self.max_utt_len = self.cfg_dic['max_utt_len']
        self.max_per_len = self.cfg_dic['max_per_len']
        self.max_per_line = self.cfg_dic['max_per_line']
        self.max_per_words = self.cfg_dic['max_per_words']
        self.num_layer = self.cfg_dic['num_layer']
        self.hops = self.cfg_dic['hops']
        self.use_copy = self.cfg_dic['use_copy']
        self.op = self.cfg_dic['op']
        self.grad_clip = self.cfg_dic['grad_clip']
        self.init_w = self.cfg_dic['init_w']
        self.batch_size = self.cfg_dic['batch_size']
        self.init_lr = self.cfg_dic['init_lr']
        self.lr_hold = self.cfg_dic['lr_hold']
        self.lr_decay = self.cfg_dic['lr_decay']
        self.keep_prob = self.cfg_dic['keep_prob']
        self.improve_threshold = self.cfg_dic['improve_threshold']
        self.patient_increase = self.cfg_dic['patient_increase']
        self.early_stop = self.cfg_dic['early_stop']
        self.max_epoch = self.cfg_dic['max_epoch']
        self.grad_noise = self.cfg_dic['grad_noise']
        self.temperature = self.cfg_dic['temperature']
        self.test_samples = self.cfg_dic['test_samples']
        self.perw_weight = self.cfg_dic['perw_weight']
        self.othw_weight = self.cfg_dic['othw_weight']
        self.balance_factor = self.cfg_dic['balance_factor']
        self.test_batchsize = self.cfg_dic['test_batchsize']
