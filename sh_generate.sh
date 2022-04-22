python run_fairseq_generate.py \
./processed \
--path ./ckpt/checkpoint_best.pt \
--user-dir ./ProphetNet_Multi/prophetnet \
--task translation_prophetnet \
--batch-size 80 \
--gen_subset test \
--beam 5 \
--num-workers 4 \
--no-repeat-ngram-size 3\
--lenpen 1.5 \
--sacrebleu \
--skip-invalid-sze-inputs-valid-test 2>&1 > ./fairseq_outputs.txt