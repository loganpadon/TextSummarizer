import matplotlib.pyplot as plt
import numpy as np
import statistics

log = open("summary_log_2019-04-26_12-09-43.txt", encoding='utf-8')

line_number = 0
# Number of fields
FIELDS = 8
# to help sort entries
i = 0
word_counts = []
similarity = []
subjects = []
sub_sim = [[]]
topics = []
top_sim = [[]]

for line in log:
    line_number += 1

    if line_number % FIELDS == 3:
        for x in word_counts:
            if x > float(line[line.index(':')+2:]):
                break
            i+=1

        word_counts.insert(i, float(line[line.index(':')+2:]))
    elif line_number % FIELDS == 4:
        similarity.insert(i, float(line[line.index(':')+2:]))
    elif line_number % FIELDS == 0:
        for sub in line[line.index(':')+2:].split(","):
            if sub not in subjects:
                subjects.append(sub)
                sub_sim.append([])
            sub_sim[subjects.index(sub)].append(similarity[i])
    elif line_number % FIELDS == 7:
        for top in line[line.index(':')+2:].split(","):
            if top not in topics:
                topics.append(top)
                top_sim.append([])
            top_sim[topics.index(top)].append(similarity[i])
    elif line_number % FIELDS == 1:
        i = 0

sub_sim = sub_sim[:-1]
top_sim = top_sim[:-1]

top_sim.remove(top_sim[topics.index('\n')])
topics.remove('\n')

# filter off entries with small samples
i = 0
while i < len(sub_sim):
    if len(sub_sim[i]) < 10:
        subjects.remove(subjects[sub_sim.index(sub_sim[i])])
        sub_sim.remove(sub_sim[i])
    else:
        i+=1

i = 0
while i < len(top_sim):
    if len(top_sim[i]) < 5:
        topics.remove(topics[top_sim.index(top_sim[i])])
        top_sim.remove(top_sim[i])
    else:
        i+=1

# shortening long labels
# its an L for labels. not the number one
for l in range(len(topics)):
    if len(topics[l]) > 20:
        topics[l] = topics[l][:20] + '...'

plt.plot(word_counts, similarity, color='blue')
plt.show()

y_pos = np.arange(len(subjects))
avg = []
for x in sub_sim:
    avg.append(statistics.mean(x))
plt.bar(y_pos, avg, align='center', alpha=0.5)
plt.xticks(y_pos, subjects)
plt.show()

y_pos = np.arange(len(topics))
avg = []
for x in top_sim:
    avg.append(statistics.mean(x))
plt.bar(y_pos, avg, align='center', alpha=0.5)
plt.xticks(y_pos, topics)
plt.show()
