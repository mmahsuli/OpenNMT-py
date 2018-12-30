# Author: Mohammad Mahdi Mahsuli
# Splitting dataset with multi-reference support
# input: 
#	src and tgt lines of train
#	src and 4 reference lines of test
# output:
#	src and tgt lines of train with len(tgt)<=max_train_tgt_len, as training data
#	src and 4 tgt references of test if for any tgt in references: len(tgt)>max_train_tgt_len, as test data

#import statistics as s
base_loc = ''#'../../data/verbmobil-enfa/'
max_train_tgt_len = 10
print('Splitting the dataset (multi-ref)...')
print('maximum target length in training dataset: '+str(max_train_tgt_len))
f1 = open(base_loc+'train.en', 'r')
f2 = open(base_loc+'train.fa', 'r')
f3 = open(base_loc+'test.en', 'r')
ref0 = open(base_loc+'ref0', 'r')
ref1 = open(base_loc+'ref1', 'r')
ref2 = open(base_loc+'ref2', 'r')
ref3 = open(base_loc+'ref3', 'r')

o1 = open(base_loc+'train-'+str(max_train_tgt_len)+'.en', 'w')
o2 = open(base_loc+'train-'+str(max_train_tgt_len)+'.fa', 'w')
o3 = open(base_loc+'test-'+str(max_train_tgt_len)+'.en', 'w')
outref0 = open(base_loc+'ref-'+str(max_train_tgt_len)+'-0', 'w')
outref1 = open(base_loc+'ref-'+str(max_train_tgt_len)+'-1', 'w')
outref2 = open(base_loc+'ref-'+str(max_train_tgt_len)+'-2', 'w')
outref3 = open(base_loc+'ref-'+str(max_train_tgt_len)+'-3', 'w')

while True:
	train_src_line = f1.readline() # train src
	if not train_src_line:
		break;	
	train_tgt_line = f2.readline() # train tgt
	if len(train_tgt_line.split()) <= max_train_tgt_len: # write to train
		o1.write(train_src_line)
		o2.write(train_tgt_line)

while True:
	ref0_line = ref0.readline()
	ref1_line = ref1.readline()
	ref2_line = ref2.readline()
	ref3_line = ref3.readline()
	test_src_line=f3.readline()		
	if not ref0_line:
		break;
	elif len(ref0_line.split()) > max_train_tgt_len or \
			len(ref1_line.split()) > max_train_tgt_len or \
			len(ref2_line.split()) > max_train_tgt_len or \
			len(ref3_line.split()) > max_train_tgt_len: #write to test
		'''a=[]
		a.append(len(ref0_line.split()))	
		a.append(len(ref1_line.split()))		
		a.append(len(ref2_line.split()))		
		a.append(len(ref3_line.split()))			
		print(s.mean(a))'''
		o3.write(test_src_line)
		outref0.write(ref0_line)
		outref1.write(ref1_line)
		outref2.write(ref2_line)
		outref3.write(ref3_line)
f1.close()
f2.close()
f3.close()
ref0.close()
ref1.close()
ref2.close()
ref3.close()
o1.close()
o2.close()
o3.close()
outref0.close()
outref1.close()
outref2.close()
outref3.close()
print('Done!')
