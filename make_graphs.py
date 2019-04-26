import matplotlib.pyplot as plt
import numpy as np
import statistics

log = open("summary_log_2019-04-25_22-40-21.txt", encoding='utf-8')

line_number = 0
# Number of fields
FIELDS = 9
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
    elif line_number % FIELDS == 7:
        similarity.insert(i, float(line[line.index(':')+2:]))
#    elif line_number % FIELDS == 4:
#        for sub in line[line.index(':')+2:].split(","):
#            if sub not in subjects:
#                subjects.append(sub)
#                sub_sim.append([])
#            sub_sim[subjects.index(sub)].append(similarity[i])
#    elif line_number % FIELDS == 5:
#        for top in line[line.index(':')+2:].split(","):
#            if top not in topics:
#                topics.append(top)
#                top_sim.append([])
#            top_sim[subjects.index(top)].append(similarity[i])
    else:
        i = 0

log.seek(0)
i = 0

# to get all the data until log file is updated
for line in log:
    line_number += 1

    if line_number % FIELDS == 3:
        for x in word_counts:
            if x >= float(line[line.index(':')+2:]):
                break
            i+=1

#        word_counts.insert(i, float(line[line.index(':')+2:]))
#    elif line_number % FIELDS == 7:
#        similarity.insert(i, float(line[line.index(':')+2:]))
    elif line_number % FIELDS == 4:
        for sub in line[line.index(':')+2:].split(","):
            if sub not in subjects:
                subjects.append(sub)
                sub_sim.append([])
            sub_sim[subjects.index(sub)].append(similarity[i])
    elif line_number % FIELDS == 5:
        for top in line[line.index(':')+2:].split(","):
            if top not in topics:
                topics.append(top)
                top_sim.append([])
            top_sim[topics.index(top)].append(similarity[i])
    else:
        i = 0
# end of extra stuff. dont forget to uncomment earlier lines
sub_sim = sub_sim[:-1]
top_sim = top_sim[:-1]
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
