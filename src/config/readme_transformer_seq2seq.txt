Download squad-split1: https://drive.google.com/open?id=1k64SgiUrhJ_dhM6-5xMzxYL4MTph49G2

1) Install OpenNMT-py from pip:

pip install OpenNMT-py

2) Install PyTorch 1.2 with Python 3

3) Proprocess the data

onmt_preprocess -train_src data/src-train.txt -train_tgt data/tgt-train.txt -valid_src data/src-val.txt -valid_tgt data/tgt-val.txt -save_data data/demo -dynamic_dict

4) Train the model

python train.py -data data/demo -save_model demo-model -layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8   -encoder_type transformer -decoder_type transformer -position_encoding -train_steps 10000000  -max_generator_batches 2 -dropout 0.1   -batch_size 60 -batch_type tokens -normalization tokens  -accum_count 2   -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2    -max_grad_norm 0 -param_init 0  -param_init_glorot -label_smoothing 0.1 -valid_steps 5000 -save_checkpoint_steps 5000 -gpu_ranks 0 --copy_attn

5) Predict

onmt_translate -model demo-model -src src-test.txt -output pred_transformer_copy_step_XXXX.txt -replace_unk --gpu 0

Note: Now you have a model which you can use to predict on new data. We do this by running beam search. This will output predictions into pred.txt.

You can follow the same way to run the model on squad-split 2. The download link to split 2 is https://drive.google.com/open?id=16oajrs_UmSPjiFyIO4mz03xgxe0zOvhW
