import csv
import numpy as np
import matplotlib.pyplot as plt

with open('Paper2_allsest_accuracy_Twitter.csv', 'r', newline='') as handle:
    reader = csv.reader(handle)
    accuracy_twitter = []
    for row in reader:
        accuracy_twitter.append(float(row[0]))

with open('Paper2_allsest_accuracy_Google.csv', 'r', newline='') as handle:
    reader = csv.reader(handle)
    accuracy_Google = []
    for row in reader:
        accuracy_Google.append(float(row[0]))

plt.figure(1)
x = np.linspace(1, len(accuracy_Google), len(accuracy_Google))
plt.plot(x, accuracy_twitter, label='Twitter word2vec', color='r')
plt.plot(x, accuracy_Google, label='Google word2vec', color='b')

plt.xticks(np.arange(1, len(accuracy_Google)+1, 2))
#plt.yticks(np.arange(94, 96, 1))
plt.gca().grid(b=False, which='both', axis='y')
plt.xlabel('Training set number')
plt.ylabel('Test accuracy (%)')
plt.legend(prop={'size': 10})
plt.savefig('Paper2_allsets_accuracy', dpi=600)
#plt.show()

print('Average accuracy of Twitter: ', np.mean(np.array(accuracy_twitter)))
print('Average accuracy of Google: ', np.mean(np.array(accuracy_Google)))
print('Max accuracy of Twitter {} and index {}'.format(np.max(np.array(accuracy_twitter)),
                                                       np.argmax(np.array(accuracy_twitter))))
print('Max accuracy of Google {} and index {} '.format(np.max(np.array(accuracy_Google)),
                                                       np.argmax(np.array(accuracy_twitter))))

a = 2