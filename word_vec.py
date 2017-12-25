# coding=utf-8
# author : liangchen
# decription :  class line_generator yield one line at a time
# 			 :  class word_vector to train all words in one directory path,generate word vectors


import gensim.models.word2vec as w2v
import logging
import numpy as np

import os
from multiprocessing import cpu_count
from datetime import datetime
import itertools

import split_word

TARGET_NEWS_ALLINONE = '/home/lc/ht_work/data/allnews_seg.txt'

STOP_WORD_FILE = '/home/lc/ht_work/hexun_word2vec/stopwords_wz.txt'
USER_DICT_FILE = '/home/lc/ht_work/hexun_word2vec/userdict_wz.txt'

MODEL_PATH = '/home/lc/ht_work/data/news.model'
VECTORS_PATH = '/home/lc/ht_work/data/news.vectors.bin'

class line_generator():
	def __init__(self,srcdir):
		self.srcdir = srcdir

	def __iter__(self):
		for fname in os.listdir(self.srcdir):
			fullpath = os.path.join(self.srcdir,fname)
			with w2v.utils.smart_open(fullpath) as fin:
				for line in itertools.islice(fin, None):
					wordline = line.split()
					i = 0
					while i < len(wordline):
						yield line[i: i + 10000]
						i += 10000


class word_vector():
	def __init__(self,linesrc,modelpath=None,retrain=True):
		self.lines = linesrc

		#self.sentences = w2v.LineSentence(self.lines)
		#self.sentences = self.linesrc

		if modelpath:
			self.model = w2v.Word2Vec.load(modelpath)
		else:
			self.model = None
		self.retrain = retrain

	def train_model(self):
		if not self.retrain:
			return self.model

		cpu_num = cpu_count()
		print u'time: %s ==> 模型训练开始，使用%d核\n' % (datetime.now(), cpu_num)
		vector_size = 300
		skip_gram = 1
		hierarch_sm = 1
		negative_sample = 0
		context_window = 5
		word_min_count = 5
		print u'time: %s ==> 模型参数： Vector size %d\tSkip-gram %d\tHierarchical softmax %d\tNeg Sampling %d\tWindows %d\tMin_count %d\n' % (datetime.now(), \
																																		 vector_size,\
																																		 skip_gram, \
																																		 hierarch_sm, \
																																		 negative_sample, \
																																		 context_window, \
																																		  word_min_count)
		#self.model = w2v.Word2Vec(self.lines, size=50, sg=0, hs=1, negative=0, window=5,min_count=5, workers=cpu_num)
		self.model = w2v.Word2Vec(w2v.LineSentence(self.lines), size=vector_size, sg=skip_gram, hs=hierarch_sm, negative=negative_sample,\
								  window=context_window, min_count=word_min_count, workers=cpu_num)
		'''
		for fname in os.listdir(self.lines):
			fullpath = os.path.join(self.lines, fname)
			if first_flag:
				self.model = w2v.Word2Vec(w2v.LineSentence(fullpath), size=50, sg=0, hs=1, negative=0, window=5,
									  min_count=5, workers=cpu_num)
				first_flag = 0
			else:
				old_corpus_count = self.model.corpus_count
				self.model.build_vocab(w2v.LineSentence(fullpath),update=True)
				new_example_count = self.model.corpus_count - old_corpus_count
				self.model.train(w2v.LineSentence(fullpath),total_examples=new_example_count,epochs=self.model.iter)
		'''

		print u'time: %s ==> 模型训练结束，使用%d核\n' % (datetime.now(), cpu_num)

		return self.model

	def update_model(self,newfile):
		if not self.model:
			return None

		if not os.path.isfile(newfile):
			return self.model

		with open(newfile,'r') as f:
			train_word_count = self.model.train(w2v.LineSentence(f))
			print u'time: %s ==> 文件%s更新模型结束，更新%d个词\n' % (datetime.now(), newfile, train_word_count)

		return self.model

	def get_vecs(self):
		if self.model:
			return self.model.wv

		return None

	def get_word_vec(self,word):
		if not self.model:
			return None

		return self.model[word]

	def save_model(self,path):
		if path and self.model:
			self.model.save(path)

		return True

	def load_model(self,path):
		if path:
			self.model = w2v.Word2Vec.load(path)
		return True

	def save_vecs(self,path):
		if path and self.model:
			self.model.wv.save_word2vec_format(path, binary=False)

		return True

	def model_test(self):
		print '胡锦涛 习近平 相似度: %f ' % self.model.similarity('胡锦涛'.decode('utf-8'), '习近平'.decode('utf-8'))
		for k in self.model.most_similar('习近平'.decode('utf-8')):
			print k[0], k[1]
		#print self.model.similarity('赵敏'.decode('utf-8'), '周芷若'.decode('utf-8'))
		#print self.model.similarity('赵敏'.decode('utf-8'), '韦一笑'.decode('utf-8'))
		print '==' * 30
		for k in self.model.similar_by_word('债券'.decode('utf-8')):
			print k[0], k[1]

		#print self.model.wv['财经'.decode('utf-8')]
		return



#class_test
if __name__=='__main__':
	#ds = split_word.doc_splitter(SRC_XWLB_DIR,TARGET_XWLB_DIR,SRC_XWLB_ALLINONE,TARGET_XWLB_ALLINONE,STOP_WORD_FILE,USER_DICT_FILE,True)
	#result_filepath = ds.split_all()

	#word_v = word_vector(line_generator('/home/lc/ht_work/ML/new_txt/'),None,True)
	result_filepath = TARGET_NEWS_ALLINONE
	word_v = word_vector(result_filepath, None, True)
	#word_v = word_vector(None,MODEL_PATH,True)
	word_v.train_model()

	word_v.save_model(MODEL_PATH)

	#word_v.load_model('/home/lc/ht_work/data/xwlb_txt/all_vecs/news.model')
	word_v.save_vecs(VECTORS_PATH)

	word_v.model_test()

