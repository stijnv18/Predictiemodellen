import csv

# Define the new headers
new_headers = ['Local Authority Code', 'Standard or Time of Use', 'Date Time', 'Kilowatt Hour per Half Hour']

# Open the original CSV file in read mode
with open('CC_LCL-FullData.csv', 'r') as infile:
    # Open a new CSV file in write mode
    with open('updated.csv', 'w', newline='') as outfile:
        # Create a CSV reader
        reader = csv.reader(infile)
        # Create a CSV writer
        writer = csv.writer(outfile)
        # Skip the original header
        next(reader)
        # Write the new header
        writer.writerow(new_headers)
        # Copy the rest of the data
        for row in reader:
            writer.writerow(row)