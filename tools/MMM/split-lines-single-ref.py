# Author: Mohammad Mahdi Mahsuli
# Splitting dataset with single-reference support
# input: 
#	src and tgt lines of train
# output:
#	src and tgt lines of train with len(tgt)<=max_train_tgt_len, as training data
#	src and tgt lines of train with len(tgt)>max_train_tgt_len, as test data

base_loc = ''#'../../data/verbmobil-enfa/'
max_train_tgt_len = 30
print('Splitting the dataset (single-ref)...')
print('maximum target length in training dataset: '+str(max_train_tgt_len))
f1 = open(base_loc+'train.en', 'r')
f2 = open(base_loc+'train.fa', 'r')
o1 = open(base_loc+'train-'+str(max_train_tgt_len)+'.en', 'w')
o2 = open(base_loc+'train-'+str(max_train_tgt_len)+'.fa', 'w')
o3 = open(base_loc+'test-'+str(max_train_tgt_len)+'.en', 'w')
o4 = open(base_loc+'test-'+str(max_train_tgt_len)+'.fa', 'w')

while True:
	src_line = f1.readline() # src
	if not src_line:
		break;	
	tgt_line = f2.readline() # tgt
	#print(len(tgt_line.split()))
	if len(tgt_line.split())<=max_train_tgt_len: # write to train
		o1.write(src_line)
		o2.write(tgt_line)
	else: #write to test
		o3.write(src_line)
		o4.write(tgt_line)
f1.close()
f2.close()
o1.close()
o2.close()
o3.close()
o4.close()
print('Done!')
