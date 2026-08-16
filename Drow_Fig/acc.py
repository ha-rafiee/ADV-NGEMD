import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

# Data
categories = ['Training_ACC', 'Validation_ACC']
pe_values = [87.1, 75.2]  # Replace the first value with actual PE before attack if known

# Plotting
plt.figure(figsize=(8,6))
bars = plt.bar(categories, pe_values, color=['blue', 'yellow'])
plt.ylim(0, 100)
plt.ylabel('Accuracy (%)')
plt.title('Accuracy related to Training and Validation Data')

# Adding the percentage labels on top of the bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, height, f'{height}%', ha='center', va='bottom')

plt.show()
