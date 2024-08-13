
[ -z "${seed}" ] && seed=0
[ -z "${threads}" ] && threads=10
[ -z "${out_dir}" ] && out_dir="./res/seed${seed}_threads${threads}/"

df_filename="${out_dir}p2rank_input_protein_remove_extra_chains_10A_list.ds"

./p2rank_2.4.1/prank predict $df_filename -o $out_dir -visualizations 0 -threads $threads -seed $seed