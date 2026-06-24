python -u ../openevolve-run.py initial_program.py ../evaluator.py --config config.yaml --output openevolve_output 2>&1 | tee openevolve_output_train.log
python ../openevolve-run.py initial_program.py ../evaluator.py --config evolved_config.yaml --output ${BENCHMARKS_args}_openevolve_output
