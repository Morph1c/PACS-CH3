import matplotlib.pyplot as plt
import numpy as np

# Load the data from the text file
data = np.loadtxt('../plots/times.txt')


# Split the data into x and y values
n_values = data[:, 0]
times = data[:, 1]

# Create the plot
plt.plot(n_values, times, marker='o')

# Add labels and title
plt.xlabel('n')
plt.ylabel('Time')
plt.title('Execution time as a function of n')

# Save the plot
plt.savefig('../plots/execution_times.png')

