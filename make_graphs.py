import matplotlib.pyplot as plt

log = open("summary_log_2019-04-24_22-18-36.txt", encoding='utf-8')

line_number = 0
# Number of fields
FIELDS = 9
# to help sort entries
i = 0
word_counts = []
Spacy_Similarity = []
Doc2Vec_Similarity = []
Jaccard_Similarity = []
TFIDF_Similarity = []
avg = []

for line in log:
    line_number += 1

    if line_number > 755:
        break

    if line_number % FIELDS == 3:
        for x in word_counts:
            if x > float(line[line.index(':')+2:]):
                break
            i+=1

        word_counts.insert(i, float(line[line.index(':')+2:]))
    elif line_number % FIELDS == 4:
        Spacy_Similarity.insert(i, float(line[line.index(':')+2:]))
    elif line_number % FIELDS == 5:
        Doc2Vec_Similarity.insert(i, float(line[line.index(':')+2:]))
    elif line_number % FIELDS == 6:
        Jaccard_Similarity.insert(i, float(line[line.index(':')+2:]))
    elif line_number % FIELDS == 7:
        TFIDF_Similarity.insert(i, float(line[line.index(':')+2:]))
    elif line_number % FIELDS == 8:
        total = Spacy_Similarity[i]
        total += Doc2Vec_Similarity[i]
        total += Jaccard_Similarity[i]
        total += TFIDF_Similarity[i]
        avg.insert(i, total/4)
    else:
        i = 0

plt.plot(word_counts, Spacy_Similarity, color='blue')
plt.plot(word_counts, Doc2Vec_Similarity, color='orange')
plt.plot(word_counts, Jaccard_Similarity, color='green')
plt.plot(word_counts, TFIDF_Similarity, color='red')
plt.plot(word_counts, avg, color='black')
plt.show()
