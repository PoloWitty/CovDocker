
[ -z "${out_dir}" ] && out_dir='./fpocket_res/'
df_filename="${out_dir}fpocket_input_protein_remove_extra_chains_10A_list.ds"

# Count the number of lines in the file
total_lines=$(wc -l < "$df_filename")

# Initialize counter
counter=0

# Use cat to read lines and process them
cat "$df_filename" | while IFS= read -r line; do
    # Process the line with fpocket
    fpocket -f "$line"

    # Increment and display the counter
    ((counter++))
    echo "Processed $counter of $total_lines"
done

echo "Processing complete!"
