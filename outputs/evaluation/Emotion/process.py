# Open the input TSV file
with open('ExplainZeroShotGPT_gpt4_corrected_by_claude3_corrected.tsv', 'r') as infile:
    # Open a new file to write the first two columns
    with open('ExplainZeroShotGPT_gpt4_corrected_by_claude3_corrected_submit.tsv', 'w') as outfile:
        # Iterate through each line in the input file
        for line in infile:
            # Split the line by tabs
            columns = line.strip().split('\t')
            # Write the first two columns to the output file
            outfile.write('\t'.join(columns[:2]) + '\n')
