# download data from PDB for covbinderInPDB
python utils/process_covbinderInPDB.py --save_dir ./data/covbinderInPDB/pdb/ --dataset_info_filename ./data/covbinderInPDB/CovBinderInPDB_2022Q4_AllRecords.csv

#  get the pdb2het.csv file from the website
python utils/pdb2het.py
mv pdb2het.csv ./data/processed/

# split the complexes into seperate bonded protein and bonded ligand files
python utils/preprocess_pdb_file.py --covpdb_complexes_path ./data/covpdb/CovPDB_complexes/ --covbinderInPDB_complexes_path ./data/covbinderInPDB/pdb/ --covbinderInPDB_info_filename ./data/covbinderInPDB/CovBinderInPDB_2022Q4_AllRecords.csv --save_dir ./data/processed/bonded --dataset_info_filename ./data/processed/examples.csv --drop_more_than_one_bond 

# cut pocket 
python utils/cut_pocket.py --dataset_info_filename ./data/processed/examples.csv --cut_off 10


# generate the dataset for position prediction
# python utils/prepare_position_data.py --dataset_info_file ./data/processed/dataset.filtered.random_split.csv --save_dir ./data/processed/dataset/position/ # use unimol_tools to extract ligand feature
python utils/prepare_reactive_site_data.py --dataset_info_file ./data/processed/dataset.filtered.random_split.csv --save_dir ./data/processed/dataset/position/

# prepare reaction data
python utils/prepare_reaction_data.py --addtional_aa_rxn_filename ''

# reaction prediction
pushd Chemformer
bash example_scripts/fine_tune.sh
bash example_scripts/predict.sh
popd

# prepare docking data
python utils/prepare_docking_data.py --max_len 1020 --dataset_info_file ./data/processed/dataset.filtered.random_split.csv

# drop failed pdb
python utils/drop_failed_pdb.py --dataset_info_filename ./data/processed/dataset.csv 

# split the dataset into train, valid and test
python utils/split_dataset.py --dataset_info_filename ./data/processed/dataset.filtered.csv --save_dir ./data/processed/dataset/


# get unseen test data
python utils/prepare_reactive_site_data.py --dataset_info_file ./data/processed/dataset.filtered.unseen.csv --save_dir ./data/processed/dataset/reactive_site/ --only-unseen 1
python utils/prepare_docking_data.py --max_len 1020 --dataset_info_file ./data/processed/dataset.unseen.csv --save_dir ./data/processed/dataset/docking/ --only-unseen 1
# for reaction data see cov_reaction_pred_baselines/ReactionT5/forward_reaction_prediction/my_prepare_input.py