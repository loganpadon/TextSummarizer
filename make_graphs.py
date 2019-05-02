import sys
import matplotlib.pyplot as plt
import statistics
import numpy as np

colors = ['blue', 'green', 'red', 'yellow', 'black', 'purple', 'pink']

def parse_file(filename):
    with open(filename, "r", encoding="utf8") as file_lines:
        results = []
        entry = {}
        entry["methods"] = {}
        for line in file_lines:
            if line[0] == "-":
                if entry:
                    results.append(entry)
                entry = {}
                entry["methods"] = {}
            else:
                if len(line.split(":")) != 2:
                    continue
                key, value = line.strip().split(":")
                if key == "Word count":
                    entry[key] = int(value)
                elif key == "Topics" or key == "Types":
                    entry[key] = value.split(",")
                elif "similarity" in key:
                    method, _ = key.split()
                    entry["methods"][method] = float(value)
        return results   

def get_methods(results):
    methods = {}
    for entry in results:
        for method, sim in entry["methods"].items():
            if not method in methods.keys():
                methods[method] = {}
                methods[method]["sims"] = []
                methods[method]["word_counts"] = []
                methods[method]["topics"] = {}
                methods[method]["types"] = {}
            methods[method]["sims"].append(sim)
            methods[method]["word_counts"].append(entry["Word count"])
            if "Topics" in entry.keys():
                for topic in entry["Topics"]:
                    if topic not in methods[method]["topics"].keys():
                        methods[method]["topics"][topic] = []
                    methods[method]["topics"][topic].append(sim)
            if "Types" in entry.keys():
                for typ in entry["Types"]:
                    if typ not in methods[method]["types"].keys():
                        methods[method]["types"][typ] = []
                    methods[method]["types"][typ].append(sim)
    return methods

def high_pass_filter(methods):
    topics_to_delete = {}
    types_to_delete = {}
    for method, vals in methods.items():
        topics_to_delete[method] = []
        types_to_delete[method] = []
        for topic, sims in vals["topics"].items():
            if len(sims) < 5:
                topics_to_delete[method].append(topic)
        for typ, sims in vals["types"].items():
            if len(sims) < 5:
                types_to_delete[method].append(typ)
    for method, topics in topics_to_delete.items():
        for topic in topics:
            del methods[method]["topics"][topic]
    for method, types in types_to_delete.items():
        for typ in types:
            del methods[method]["types"][typ]

def plot_word_counts(methods):
    i = 0
    for method, vals in methods.items():
        x = vals["word_counts"]
        y = vals["sims"]
        sorted_x, sorted_y = zip(*sorted(zip(x, y)))
        plt.plot(sorted_x, sorted_y, color=colors[i])
        plt.xlabel('Word Count')
        plt.ylabel('Similarity')
        plt.title(method + " Similarity vs Word Count")
        plt.show()
        i += 1
    i = 0
    for method, vals in methods.items():
        x = vals["word_counts"]
        y = vals["sims"]
        sorted_x, sorted_y = zip(*sorted(zip(x, y)))
        plt.plot(sorted_x, sorted_y, color=colors[i])
        i += 1
    plt.xlabel('Word Count')
    plt.ylabel('Similarity')
    plt.title('Combined Similarity vs Word Count')
    plt.show()

def plot_topics(methods):
    for method, vals in methods.items():
        topics = vals["topics"].keys()
        y_pos = np.arange(len(topics))
        avg = []
        for topic in topics:
            avg.append(statistics.mean(vals["topics"][topic]))
        plt.bar(y_pos, avg, align='center', alpha=0.5)
        plt.xticks(y_pos, topics, rotation=90)
        plt.ylabel('Similarity')
        plt.title(method + " Summary Similarity for Topics")
        plt.show()

def plot_types(methods):
    for method, vals in methods.items():
        types = vals["types"].keys()
        y_pos = np.arange(len(types))
        avg = []
        for typ in types:
            avg.append(statistics.mean(vals["types"][typ]))
        plt.bar(y_pos, avg, align='center', alpha=0.5)
        plt.xticks(y_pos, types, rotation=90)
        plt.ylabel('Similarity')
        plt.title(method + " Summary Similarity for Types")
        plt.show()

def get_topics(results):
    topics = {}
    for entry in results:
        if "Topics" not in entry.keys():
            continue
        for topic in entry["Topics"]:
            if topic not in topics.keys():
                topics[topic] = {}
            for method, sim in entry["methods"].items():
                if method not in topics[topic].keys():
                    topics[topic][method] = []
                topics[topic][method].append(sim)
    topics_to_delete = []
    for topic, methods in topics.items():
        max_size = 0
        for method, sims in methods.items():
            if len(sims) > max_size:
                max_size = len(sims)
        if max_size < 5:
            topics_to_delete.append(topic)
    for topic in topics_to_delete:
        del topics[topic]
    return topics


def run_analysis(methods, filename):
    with open(filename,"w+",encoding="utf8") as analysis:
        for method, vals in methods.items():
            sims = vals["sims"]

            analysis.write(method + " Summary\n")
            analysis.write("Mean Similarity: " + str(statistics.mean(sims)) + "\n")
            analysis.write("Median Similarity : " + str(statistics.median(sims)) + "\n")
            analysis.write("Min Similarity: " + str(min(sims)) + "\n")
            analysis.write("Max Similarity: " + str(max(sims)) + "\n\n")

def main():
    if len(sys.argv) != 2:
        print("Usage: python make_graphs.py [filename]")
        exit()
    filename = sys.argv[1]
    analysis_filename = filename.split(".")[-2]+"_analysis.txt"
    results = parse_file(filename)
    methods = get_methods(results)
    high_pass_filter(methods)
    run_analysis(methods, analysis_filename)
    plot_word_counts(methods)
    plot_topics(methods)
    plot_types(methods)

if __name__=="__main__":
    main()