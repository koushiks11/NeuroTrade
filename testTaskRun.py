import datetime

# Function to read the last run time from file
def read_last_run_time(filename):
    try:
        with open(filename, 'r') as file:
            last_run_time = file.read()
            return last_run_time
    except FileNotFoundError:
        return None

# Function to append current time to file
def append_current_time(filename):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(filename, 'a') as file:
        file.write(current_time + '\n')  # Append current time to a new line

# File to store last run times
filename = "last_run_times.txt"

# Read the last run time
last_run_time = read_last_run_time(filename)

if last_run_time:
    print("Last run times:\n" + last_run_time)
else:
    print("No previous run recorded.")

# Append current time to file
append_current_time(filename)
