# coding=utf-8
# author : liangchen
# decription : class doc_splitter to split all words in one directory path,generate new files(with segmented words in)

import os
from multiprocessing import cpu_count
import re
from datetime import datetime

from pathlib import Path
import jieba
import json

class doc_splitter():
	def __init__(self,srcdir,targetdir,srcfile,targetfile,stopwordfile=None,userdict=None,parallel=True):
		self.srcdir = srcdir
		self.srcfile = srcfile

		self.tagdir = targetdir
		self.tagfile = targetfile

		self.stopwordfile = stopwordfile  #dict
		self.swlist = None
		self.parallel = parallel
		self.userdict = userdict

		self.split_count = 0

		jieba.initialize()
		if self.userdict:
			jieba.load_userdict(userdict)

		if self.parallel:
			jieba.enable_parallel(cpu_count())

	def utf8_one_doc(self,filepath):
		if not filepath:
			return

		print u'time: %s ==> 转换文件%s为utf-8格式\n' % (datetime.now(),filepath)
		filepath_utf8 = '/tmp' + os.path.sep + os.path.splitext(os.path.basename(filepath))[0] + '_utf8' + '.txt'

		if os.path.exists(filepath_utf8):
			print u'time: %s ==> 文件%s已经转换为%s\n' % (datetime.now(), filepath,filepath_utf8)
			return filepath_utf8

		with open(str(filepath),'r') as fr:
			with open(filepath_utf8,'w') as fw:
				line = fr.readline()
				while line:
					if line == '\r\n':
						line = fr.readline()
						continue
					newline = line.decode('GB18030').encode('utf-8')
					print newline
					print >> fw, newline
					line = fr.readline()

		print u'time: %s ==> 文件%s已经转换为%s\n' % (datetime.now(), filepath, filepath_utf8)
		return filepath_utf8

	def split_one_doc(self,filepath,savetoone=False):
		if not filepath:
			return

		print u'time: %s ==> 对文件%s进行分词\n' % (datetime.now(), filepath)
		#filepath_segmented = self.tagdir + os.path.sep + os.path.splitext(os.path.basename(filepath))[0] + '_segmented' + '.txt'
		if savetoone:
			filepath_segmented = self.tagfile
		else:
			filepath_segmented = self.tagdir + os.path.sep + os.path.splitext(os.path.basename(filepath))[0] + '_segmented' + '.txt'

		self.split_count += 1

		try:
			with open(filepath,'r') as fr:
				with open(filepath_segmented,'a') as fw:
					text = fr.read()
					js_dict = json.loads(text,strict=False)

					content = js_dict['title'] + js_dict['desc']
					content_clean = re.sub("[0-9\s+\.\!\/_,$%^*()?;；:-【】><+\"\']+|[+－——！，;:：。？、~@#￥%……&*（）]+".decode("utf8"), \
										   "".decode("utf8"), content)

					word_list = list(jieba.cut(content_clean,cut_all=False,HMM=True))  # 用结巴分词，对每行内容进行分词
					if self.swlist:
						out_str = [word+' ' for word in word_list if word not in self.swlist]
					else:
						out_str = word_list
					fw.write(" ".join(out_str).strip().encode('utf-8') + '\n')  # 将分词好的结果写入到输出文件

			print u'time: %s ==> 文件%s分词结束，保存到%s\n' % (datetime.now(), filepath, filepath_segmented)
			return filepath_segmented
		except Exception,e:
			print e.message
			fullpath = '/tmp/badones/' + str(self.split_count) + '_' + os.path.splitext(os.path.basename(filepath))[0] + '_' + e.message + '.bad'
			os.system('cp %s %s' % (filepath,fullpath))
			return None


	def split_one_by_one(self):
		if self.srcdir:
			p = Path(self.srcdir)
			for txt in p.glob("**/*.json"):
				#txt_utf8 = self.utf8_one_doc(str(txt))
				#self.split_one_doc(txt_utf8)
				if str(txt) == self.srcfile:
					continue

				self.split_one_doc(str(txt),savetoone=True)

	def split_all(self):
		if self.stopwordfile and os.path.isfile(self.stopwordfile):
			self.swlist = [line.strip().decode('utf-8') for line in open(self.stopwordfile,'r')]

		if self.srcfile:
			return self.split_one_doc(self.srcfile,True)

		return None


#class test:
if __name__=='__main__':
	if not os.path.exists('/tmp/badones'):
		os.mkdir('/tmp/badones')
	ds = doc_splitter('/home/lc/ht_work/data/hexun_news/','/home/lc/ht_work/data/hexun_res/',None,'/home/lc/ht_work/data/hexun_res/allnews_seg.txt','/home/lc/ht_work/hexun_word2vec/stopwords_wz.txt','/home/lc/ht_work/hexun_word2vec/userdict_wz.txt',True)
	ds.split_all()
	ds.split_one_by_one()

