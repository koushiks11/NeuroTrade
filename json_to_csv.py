import json
import csv

def json_to_csv(input_file, output_file):
    with open(input_file, 'r') as json_file:
        data = json.load(json_file)

    # Extract keys (dates)
    dates = sorted(data.keys())
    keys = data[dates[0]].keys()

    with open(output_file, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=['Date'] + list(keys))
        writer.writeheader()
        
        for date in dates:
            row = {'Date': date}
            row.update(data[date])
            writer.writerow(row)

# Example usage:
json_to_csv('sample.json', 'output.csv')
