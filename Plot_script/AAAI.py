import matplotlib.pyplot as plt

# Data for plotting
names = ['QResVAE', 'Entroformer', 'Cheng2020', 'ELIC', 'WACNN', 'STF', 'Ours', 'Mixed', 'VVC Intra']
x = [400, 600, 1200, 800, 1000, 1100, 300, 1600, 1800]  # MACs/pixel (K)
y = [0, -4, -6, 6, 8, 2, 10, 10, -2]  # BD-Rate Reduction (%)
sizes = [34.0, 44.9, 26.5, 33.8, 75.2, 99.8, 78.8, 75.8, 0]  # Parameter size (M) - area of bubbles
colors = ['orange', 'blue', 'yellow', 'lightblue', 'cyan', 'green', 'red', 'purple', 'grey']
years = [2023, 2022, 2020, 2022, 2022, 2022, 2023, 2023, 2023]

# Create the scatter plot
plt.figure(figsize=(10, 7.5))
scatter = plt.scatter(x, y, s=[size * 20 for size in sizes], c=colors, alpha=0.5)

# Add annotations
for i, name in enumerate(names):
    if name == 'VVC Intra':
        continue  # Skip annotation for 'VVC Intra'
    elif name == 'Ours':
        plt.scatter(x[i], y[i], s=sizes[i] * 20, c=colors[i], marker='*', label=name)  # Use a star marker for 'Ours'
    else:
        plt.annotate(f'{name}\n{sizes[i]}M\n{years[i]}',
                     (x[i], y[i]),
                     textcoords="offset points",
                     xytext=(0, 10),
                     ha='center')

# Add dashed line
plt.axhline(0, color='orange', linestyle='--')

# Adjust the x-axis and y-axis limits
plt.xlim(200, 2000)
plt.ylim(-8, 12)

# Add labels and title
plt.xlabel('MACs/pixel (K)')
plt.ylabel('BD-Rate Reduction (%)')
plt.title('Comparison of Video Compression Methods')

# Show grid
plt.grid(True)

# Add legend
plt.legend()

# Display the plot
plt.show()
