python run_fairseq_interactive.py \
./processed \
--path ./ckpt/checkpoint_best.pt \
--user-dir ./ProphetNet_Multi/prophetnet \
--task translation_prophetnet \
--gen-subset test \
--beam 5 \
--num-workers 4 \
--no-repeat-ngram-size 3 \
--lenpen 1.5 \
--sacrebleu \
--skip-invalid-size-inputs-valid-test