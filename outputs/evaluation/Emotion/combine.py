def read_tsv(filename):
    data = {}
    with open(filename, 'r') as file:
        for line in file:
            columns = line.strip().split('\t')
            data[columns[0]] = columns[1]
    return data

def main():
    # Replace these filenames with your actual filenames
    file1_data = read_tsv('../../../data/exalt_test_participants/exalt_emotion_test_processed.tsv')
    file2_data = read_tsv('../../../data/exalt_test_participants/exalt_emotion_test_participants.tsv')

    with open('diff', 'r') as diff_file:
        with open('output_combined.txt', 'w') as combined_file:
            id_record = []
            for line in diff_file:
                combined_file.write(line)
                items = line.strip().split()
                if len(items) == 3:
                    id = items[1]
                    if id in id_record:
                        combined_file.write(file1_data[id] + '\n')
                        combined_file.write(file2_data[id] + '\n')
                    else:
                        id_record.append(id)
                    
                    

if __name__ == "__main__":
    main()
