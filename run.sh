python ./trainSpeakerNet.py --model ResNetSE34V2 --log_input True --nClasses 17121 --encoder_type ASP --n_mels 3 --trainfunc softmaxproto --save_path exps/exp2 --nPerSpeaker 2 --batch_size 200 --train_list /data/juicefs_speech_tts_v3/public_data/11139034/vp_corpus/train.list --train_path ''  --test_list /data/juicefs_speech_tts_v3/public_data/11139034/vp_corpus/test.list --test_path ''

