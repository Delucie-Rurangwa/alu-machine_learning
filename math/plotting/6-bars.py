#!/usr/bin/env python3
'''
Plots a stacked bar graph for the fruit matrix
'''
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4,3))

people = ['Farrah', 'Fred', 'Felicia']
x = np.arange(len(people))
width = 0.5

# Fruit data (rows): apples, bananas, oranges, peaches
apples = fruit[0]
bananas = fruit[1]
oranges = fruit[2]
peaches = fruit[3]

# Create stacked bar chart
plt.bar(x, apples, width, label='apples', color='red')
plt.bar(x, bananas, width, bottom=apples, label='bananas', color='yellow')
plt.bar(x, oranges, width, bottom=apples + bananas, label='oranges', color='#ff8000')
plt.bar(x, peaches, width, bottom=apples + bananas + oranges, label='peaches', color='#ffe5b4')

# Configure axes
plt.xticks(x, people)
plt.ylabel('Quantity of Fruit')
plt.ylim(0, 80)
plt.yticks(range(0, 81, 10))
plt.title('Number of Fruit per Person')
plt.legend()

plt.show()