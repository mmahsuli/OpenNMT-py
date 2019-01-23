#! /bin/sh
#########################################################################################
#				Experiment Automation					#
#			    Author: Mohammad Mahdi Mahsuli				#
#########################################################################################
# Parameters:										#
#########################################################################################
# model_dir: directory under OpenNMT-py root directory containing the models		#
# model_prefix: the prefix pattern of the models to use					#
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

model_dir='model'
model_prefix='iwslt-baseline-40'
openNMT_py_dir='../../'
src='data/iwslt14.tokenized.de-en/test-40.de'
tgt='data/iwslt14.tokenized.de-en/test-40.en'
multiref_tgt='data/verbmobil-enfa/ref'

file_suffix='' #'-oracle'
translation_extra_params='' #'-length_model oracle'

exp_dir='experiments'
test_single_ref=true
test_multi_ref=false

cd $openNMT_py_dir

if [ ! -d $exp_dir ]; then
    mkdir $exp_dir
fi

if [ -f $exp_dir/${model_prefix}${file_suffix}-single-bleu-stats.txt ] && [ $test_single_ref = true ]; then
    rm $exp_dir/${model_prefix}${file_suffix}-single-bleu-stats.txt
fi

if [ -f $exp_dir/${model_prefix}${file_suffix}-multi-bleu-stats.txt ] && [ $test_multi_ref = true ]; then
    rm $exp_dir/${model_prefix}${file_suffix}-multi-bleu-stats.txt
fi

for model in $(ls $model_dir/${model_prefix}_step_*.pt | sort -n -t _ -k 3); do
    iter=$(echo $model | grep -o -P '(?<=_step_).*(?=.pt)')
    trans_dir=$(echo $model | grep -o -P '(?<=model/).*(?=_step_.*)')$file_suffix
    echo '==========================='
    echo iteration: $iter
    echo '==========================='
    if [ ! -d $exp_dir/$trans_dir ]; then
        mkdir $exp_dir/$trans_dir
    fi
    output_suffix=$(echo $model | grep -o -P "(?<=${model_dir}/).*(?=.pt)")
    python3 translate.py -src $src -tgt $tgt -replace_unk -verbose -gpu 0 -model $model -output ${exp_dir}/${trans_dir}/pred-${output_suffix}${file_suffix}.txt -max_length 200 $translation_extra_params
    #single-reference BLEU calculation
    if [ $test_single_ref = true ] ; then
        bleu_details=$(perl tools/multi-bleu.perl  $tgt < ${exp_dir}/${trans_dir}/pred-${output_suffix}${file_suffix}.txt)
        echo $iter'\t'$(echo $bleu_details | grep -o -P '(?<=BLEU = )\d{2}.\d{2}(?=, )')'\t'$bleu_details >> $exp_dir/${model_prefix}${file_suffix}-single-bleu-stats.txt
    fi

    #multi-reference BLEU calculation
    if [ $test_multi_ref = true ] ; then
        bleu_details=$(perl tools/multi-bleu.perl  $multiref_tgt < ${exp_dir}/${trans_dir}/pred-${output_suffix}${file_suffix}.txt)
        echo $iter'\t'$(echo $bleu_details | grep -o -P '(?<=BLEU = )\d{2}.\d{2}(?=, )')'\t'$bleu_details >> $exp_dir/${model_prefix}${file_suffix}-multi-bleu-stats.txt
    fi
done
echo '==========================='
echo 'All Done!'
