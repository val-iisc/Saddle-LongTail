import os
path = "output"
folders = sorted(os.listdir(path))
for i in folders:
    log_file_path = os.path.join(path, i, "log.txt")
    highest_prec1 = 0.0
    save_line =""

    # Open the log file and read its contents line by line
    with open(log_file_path, 'r') as file:
        for line in file:
            # Split the line into words based on spaces
            words = line.split()

            # Find the index of 'Prec@1' in the words list
            prec1_index = words.index('Prec@1')

            # Extract the Prec@1 value and convert it to a float
            prec1_value = float(words[prec1_index + 1])

            # Update the highest_prec1 value if the current value is higher
            if prec1_value > highest_prec1:
                highest_prec1 = prec1_value
                save_line = line

    # Print the highest Prec@1 value
    print(f"{i}; Highest Prec@1 value: {highest_prec1}; {save_line}")

