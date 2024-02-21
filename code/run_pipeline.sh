#!/bin/bash
#SBATCH --job-name=triviaqa
#SBATCH --mail-type=ALL
#SBATCH --mail-user=omer_ronen@berkeley.edu
#SBATCH -o triviaqa.out #File to which standard out will be written
#SBATCH -p jsteinhardt
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=100G


run_id=$1
model=$2

cd /accounts/campus/omer_ronen/projects/semantic_uncertainty

# run generation
/accounts/campus/omer_ronen/.conda/envs/lso/bin/python code/generate.py --num_generations_per_prompt='5' --model=$model --fraction_of_data_to_use='0.1' --run_id=$run_id --temperature='0.5' --num_beams='1' --top_p='1.0'
# get semantic similarities
#/accounts/campus/omer_ronen/.conda/envs/lso/bin/python clean_generated_strings.py  --generation_model=$model --run_id=$run_id
/accounts/campus/omer_ronen/.conda/envs/lso/bin/python code/get_semantic_similarities.py --generation_model=$model --run_id=$run_id
# get likelihoods
/accounts/campus/omer_ronen/.conda/envs/lso/bin/python code/get_likelihoods.py --evaluation_model=$model --generation_model=$model --run_id=$run_id
# get prompting based uncertainty
/accounts/campus/omer_ronen/.conda/envs/lso/bin/python code/get_prompting_based_uncertainty.py --run_id_for_few_shot_prompt=$run_id --run_id_for_evaluation=$run_id
# compute confidence measure
/accounts/campus/omer_ronen/.conda/envs/lso/bin/python code/compute_confidence_measure.py --generation_model=$model --evaluation_model=$model --run_id=$run_id



