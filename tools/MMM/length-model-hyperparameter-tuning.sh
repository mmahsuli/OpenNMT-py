#! /bin/bash
#########################################################################################
#			length model hyperparameter tuning				#
#			    Author: Mohammad Mahdi Mahsuli				#
#########################################################################################
# Parameters:										#
#########################################################################################
# model_dir: directory under OpenNMT-py root directory containing the models		#
# model: model to be used for translation						#
# openNMT_py_dir: root director of OpenNMT-py						#
# src: source test file									#
# tgt: target test file									#
# multiref_tgt: prefix of multi-reference target test files				#
# file_suffix: suffix of output to address extra parameters used for translation like	#
# 			    '-oracle'. Leave it empty when there is no extra params.	#
# translation_extra_params: extra translation parameters like '-length_model oracle'.	#
# 			    Leave it empty when there is no extra params.		#
# exp_dir: directory of experiments under OpenNMT-py root directory			#
# test_single_ref [true | false]: wheather or not test using single-reference		#
# test_multi_ref [true | false]: wheather or not test using multi-reference		#
#########################################################################################

model_dir='model/iwslt-deen-30'
model='transformer-iwslt-30_step_24000.pt'
openNMT_py_dir='../../'
src='data/iwslt14.tokenized.de-en/valid-30.de'
tgt='data/iwslt14.tokenized.de-en/valid-30.en'
multiref_tgt='data/verbmobil-enfa/ref'

file_suffix='-lstm'
translation_extra_params='-length_model lstm -length_model_loc length_model_iwslt-30_ratio_epoch_13.pt'

exp_dir='experiments/hyperparameter-tuning/iwslt-30'
test_single_ref=true
test_multi_ref=false

cd $openNMT_py_dir

if [ ! -d $exp_dir ]; then
    mkdir $exp_dir
fi

if [ -f $exp_dir/${model}${file_suffix}-single-bleu-stats.txt ] && [ $test_single_ref = true ]; then
    rm $exp_dir/${model}${file_suffix}-single-bleu-stats.txt
fi

if [ -f $exp_dir/${model}${file_suffix}-multi-bleu-stats.txt ] && [ $test_multi_ref = true ]; then
    rm $exp_dir/${model}${file_suffix}-multi-bleu-stats.txt
fi


if [ $test_single_ref = true ] ; then
    echo 'a\tb\tBLEU\tdetails' >> $exp_dir/${model}${file_suffix}-single-bleu-stats.txt
fi

if [ $test_multi_ref = true ] ; then
    echo 'a\tb\tBLEU\tdetails' >> $exp_dir/${model}${file_suffix}-multi-bleu-stats.txt
fi

a_end=40
b_end=3
for a in $(seq 1 $a_end); do
    for b in $(seq 1 $b_end); do
        trans_file=${model}${file_suffix}_a${a}_b${b}
        echo '==========================='
        echo a: ${a}'\t'b: ${b}
        echo '==========================='

        python3 translate.py -src $src -tgt $tgt -replace_unk -verbose -gpu 0 -model ${model_dir}/${model} -output ${exp_dir}/pred-${trans_file}.txt -max_length 200 -length_penalty_a $a -length_penalty_b $b $translation_extra_params
        #single-reference BLEU calculation
        if [ $test_single_ref = true ] ; then
            bleu_details=$(perl tools/multi-bleu.perl  $tgt < ${exp_dir}/pred-${trans_file}.txt)
            echo $a'\t'$b'\t'$(echo $bleu_details | grep -o -P '(?<=BLEU = )\d{2}.\d{2}(?=, )')'\t'$bleu_details >> $exp_dir/${model}${file_suffix}-single-bleu-stats.txt
        fi

        #multi-reference BLEU calculation
        if [ $test_multi_ref = true ] ; then
            bleu_details=$(perl tools/multi-bleu.perl  $multiref_tgt < ${exp_dir}/pred-${file_suffix}.txt)
            echo $a'\t'$b'\t'$(echo $bleu_details | grep -o -P '(?<=BLEU = )\d{2}.\d{2}(?=, )')'\t'$bleu_details >> $exp_dir/${model}${file_suffix}-multi-bleu-stats.txt
        fi
    done
done
echo '==========================='
echo 'All Done!'
