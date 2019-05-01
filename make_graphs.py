import matplotlib.pyplot as plt
import numpy as np
import statistics

FILE = "summarizers/logs/summary_log_2019-04-30_13-44-59.txt"

log = open(FILE, encoding='utf-8')
analysis = open(FILE[:-4] + "_analysis.txt", "w", encoding='utf-8')

line_number = 0
# Number of fields
FIELDS = 14
# to help sort entries
i = 0
word_counts = []
basic_similarity = []
sim_matrix_similarity = []
text_rank_similarity = []
frequency_similarity = []
types = []
topics = []
b_typ_sim = [[]]
b_top_sim = [[]]
sm_typ_sim = [[]]
sm_top_sim = [[]]
tr_typ_sim = [[]]
tr_top_sim = [[]]
f_typ_sim = [[]]
f_top_sim = [[]]

TIME = 1
TITLE = 2
WORD_COUNT = 3
ABSTRACT = 4
BASIC_SIMILARITY = 5
BASIC_SUMMARY = 6
SIM_MATRIX_SIMILARITY = 7
SIM_MATRIX_SUMMARY = 8
TEXT_RANK_SIMILARITY = 9
TEXT_RANK_SUMMARY = 10
FREQUENCY_SIMILARITY = 11
FREQUENCY_SUMMARY = 12
TOPICS = 13
TYPES = 0

for line in log:
    line_number += 1

    if line_number % FIELDS == WORD_COUNT:
        for x in word_counts:
            if x > float(line[line.index(':')+2:]):
                break
            i+=1

        word_counts.insert(i, float(line[line.index(':')+2:]))
        
    elif line_number % FIELDS == BASIC_SIMILARITY:
        basic_similarity.insert(i, float(line[line.index(':')+2:]))
    elif line_number % FIELDS == SIM_MATRIX_SIMILARITY:
        sim_matrix_similarity.insert(i, float(line[line.index(':')+2:]))
    elif line_number % FIELDS == TEXT_RANK_SIMILARITY:
        text_rank_similarity.insert(i, float(line[line.index(':')+2:]))
    elif line_number % FIELDS == FREQUENCY_SIMILARITY:
        frequency_similarity.insert(i, float(line[line.index(':')+2:]))

    elif line_number % FIELDS == TYPES:
        for typ in line[line.index(':')+2:].split(","):
            if typ not in types:
                types.append(typ)
                b_typ_sim.append([])
                sm_typ_sim.append([])
                tr_typ_sim.append([])
                f_typ_sim.append([])
            b_typ_sim[types.index(typ)].append(basic_similarity[i])
            sm_typ_sim[types.index(typ)].append(sim_matrix_similarity[i])
            tr_typ_sim[types.index(typ)].append(text_rank_similarity[i])
            f_typ_sim[types.index(typ)].append(frequency_similarity[i])
            
    elif line_number % FIELDS == TOPICS:
        for top in line[line.index(':')+2:].split(","):
            if top not in topics:
                topics.append(top)
                b_top_sim.append([])
                sm_top_sim.append([])
                tr_top_sim.append([])
                f_top_sim.append([])
            b_top_sim[topics.index(top)].append(basic_similarity[i])
            sm_top_sim[topics.index(top)].append(sim_matrix_similarity[i])
            tr_top_sim[topics.index(top)].append(text_rank_similarity[i])
            f_top_sim[topics.index(top)].append(frequency_similarity[i])
    elif line_number % FIELDS == TIME:
        i = 0

b_typ_sim = b_typ_sim[:-1]
b_top_sim = b_top_sim[:-1]
sm_typ_sim = sm_typ_sim[:-1]
sm_top_sim = sm_top_sim[:-1]
tr_typ_sim = tr_typ_sim[:-1]
tr_top_sim = tr_top_sim[:-1]
f_typ_sim = f_typ_sim[:-1]
f_top_sim = f_top_sim[:-1]

b_top_sim.remove(b_top_sim[topics.index('\n')])
sm_top_sim.remove(sm_top_sim[topics.index('\n')])
tr_top_sim.remove(tr_top_sim[topics.index('\n')])
f_top_sim.remove(f_top_sim[topics.index('\n')])
topics.remove('\n')

# filter off entries with small samples
i = 0
while i < len(b_typ_sim):
    if len(b_typ_sim[i]) < 2:
        types.remove(types[b_typ_sim.index(b_typ_sim[i])])
        b_typ_sim.remove(b_typ_sim[i])
        sm_typ_sim.remove(sm_typ_sim[i])
        tr_typ_sim.remove(tr_typ_sim[i])
        f_typ_sim.remove(f_typ_sim[i])

    else:
        i+=1

i = 0
while i < len(b_top_sim):
    if len(b_top_sim[i]) < 5:
        topics.remove(topics[b_top_sim.index(b_top_sim[i])])
        b_top_sim.remove(b_top_sim[i])
        sm_top_sim.remove(sm_top_sim[i])
        tr_top_sim.remove(tr_top_sim[i])
        f_top_sim.remove(f_top_sim[i])

    else:
        i+=1

# shortening long labels
# its an L for labels. not the number one
for l in range(len(topics)):
    if len(topics[l]) > 20:
        topics[l] = topics[l][:20] + '...'

analysis.write('Basic Summary\n')
analysis.write('Mean Similarity: ')
analysis.write(str(statistics.mean(basic_similarity)))
analysis.write('\n')
analysis.write('Median Similarity: ')
analysis.write(str(statistics.median(basic_similarity)))
analysis.write('\n')
analysis.write('Minimum Similarity: ')
analysis.write(str(min(basic_similarity)))
analysis.write('\n')
analysis.write('Maximum Similarity: ')
analysis.write(str(max(basic_similarity)))
analysis.write('\n\n')

analysis.write('Sim_matrix Summary\n')
analysis.write('Mean Similarity: ')
analysis.write(str(statistics.mean(sim_matrix_similarity)))
analysis.write('\n')
analysis.write('Median Similarity: ')
analysis.write(str(statistics.median(sim_matrix_similarity)))
analysis.write('\n')
analysis.write('Minimum Similarity: ')
analysis.write(str(min(sim_matrix_similarity)))
analysis.write('\n')
analysis.write('Maximum Similarity: ')
analysis.write(str(max(sim_matrix_similarity)))
analysis.write('\n\n')

analysis.write('Text_rank Summary\n')
analysis.write('Mean Similarity: ')
analysis.write(str(statistics.mean(text_rank_similarity)))
analysis.write('\n')
analysis.write('Median Similarity: ')
analysis.write(str(statistics.median(text_rank_similarity)))
analysis.write('\n')
analysis.write('Minimum Similarity: ')
analysis.write(str(min(text_rank_similarity)))
analysis.write('\n')
analysis.write('Maximum Similarity: ')
analysis.write(str(max(text_rank_similarity)))
analysis.write('\n\n')

analysis.write('Frequency Summary\n')
analysis.write('Mean Similarity: ')
analysis.write(str(statistics.mean(frequency_similarity)))
analysis.write('\n')
analysis.write('Median Similarity: ')
analysis.write(str(statistics.median(frequency_similarity)))
analysis.write('\n')
analysis.write('Minimum Similarity: ')
analysis.write(str(min(frequency_similarity)))
analysis.write('\n')
analysis.write('Maximum Similarity: ')
analysis.write(str(max(frequency_similarity)))
analysis.write('\n')

plt.plot(word_counts, basic_similarity, color='blue')
plt.plot(word_counts, sim_matrix_similarity, color='green')
plt.plot(word_counts, text_rank_similarity, color='red')
plt.plot(word_counts, frequency_similarity, color='black')
plt.xlabel('Word Count')
plt.ylabel('Similarity')
plt.title('Combined Similarity vs Word Count')
plt.show()

plt.plot(word_counts, basic_similarity, color='blue')
plt.xlabel('Word Count')
plt.ylabel('Similarity')
plt.title('Basic Similarity vs Word Count')
plt.show()

plt.plot(word_counts, sim_matrix_similarity, color='green')
plt.xlabel('Word Count')
plt.ylabel('Similarity')
plt.title('Sim_matrix Similarity vs Word Count')
plt.show()

plt.plot(word_counts, text_rank_similarity, color='red')
plt.xlabel('Word Count')
plt.ylabel('Similarity')
plt.title('Text_rank Similarity vs Word Count')
plt.show()

plt.plot(word_counts, frequency_similarity, color='black')
plt.xlabel('Word Count')
plt.ylabel('Similarity')
plt.title('Frequency Similarity vs Word Count')
plt.show()

y_pos = np.arange(len(types))

avg = []
for x in b_typ_sim:
    avg.append(statistics.mean(x))
plt.bar(y_pos, avg, align='center', alpha=0.5)
plt.xticks(y_pos, types)
plt.ylabel('Similarity')
plt.title('Basic Summary Similarity for Types')
plt.show()

avg = []
for x in sm_typ_sim:
    avg.append(statistics.mean(x))
plt.bar(y_pos, avg, align='center', alpha=0.5)
plt.xticks(y_pos, types)
plt.ylabel('Similarity')
plt.title('Sim_matrix Summary Similarity for Types')
plt.show()

avg = []
for x in tr_typ_sim:
    avg.append(statistics.mean(x))
plt.bar(y_pos, avg, align='center', alpha=0.5)
plt.xticks(y_pos, types)
plt.ylabel('Similarity')
plt.title('Text_rank Summary Similarity for Types')
plt.show()

avg = []
for x in f_typ_sim:
    avg.append(statistics.mean(x))
plt.bar(y_pos, avg, align='center', alpha=0.5)
plt.xticks(y_pos, types)
plt.ylabel('Similarity')
plt.title('Frequency Summary Similarity for Types')
plt.show()


y_pos = np.arange(len(topics))

avg = []
for x in b_top_sim:
    avg.append(statistics.mean(x))
plt.bar(y_pos, avg, align='center', alpha=0.5)
plt.xticks(y_pos, topics)
plt.ylabel('Similarity')
plt.title('Basic Summary Similarity for Topics')
plt.show()

avg = []
for x in sm_top_sim:
    avg.append(statistics.mean(x))
plt.bar(y_pos, avg, align='center', alpha=0.5)
plt.xticks(y_pos, topics)
plt.ylabel('Similarity')
plt.title('Sim_matrix Summary Similarity for Topics')
plt.show()

avg = []
for x in tr_top_sim:
    avg.append(statistics.mean(x))
plt.bar(y_pos, avg, align='center', alpha=0.5)
plt.xticks(y_pos, topics)
plt.ylabel('Similarity')
plt.title('Text_rank Summary Similarity for Topics')
plt.show()

avg = []
for x in f_top_sim:
    avg.append(statistics.mean(x))
plt.bar(y_pos, avg, align='center', alpha=0.5)
plt.xticks(y_pos, topics)
plt.ylabel('Similarity')
plt.title('Frequency Summary Similarity for Topics')
plt.show()
