import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

# Data
categories = ['NGEMD', 'ADV-NGEMD']
pe_values = [3, 60]  # Replace the first value with actual PE before attack if known

# Plotting
plt.figure(figsize=(8,6))
bars = plt.bar(categories, pe_values, color=['green', 'red'])
plt.ylim(0, 100)
plt.ylabel('Miss Detection (%)')
plt.title('False Detection Probability in YeNet Steganalysis (0.4 bpp)')

# Adding the percentage labels on top of the bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, height, f'{height}%', ha='center', va='bottom')

plt.show()
