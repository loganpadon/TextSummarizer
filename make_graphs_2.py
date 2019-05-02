import sys
import matplotlib.pyplot as plt
import statistics

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
                key, value = line.split(":")
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
            methods[method]["sims"].append(sim)
            methods[method]["word_counts"].append(entry["Word count"])
    return methods

def plot_word_counts(methods):
    for method, vals in methods.items():
        x = vals["word_counts"]
        y = vals["sims"]
        sorted_x, sorted_y = zip(*sorted(zip(x, y)))
        print(sum(y) / len(y))
        plt.plot(sorted_x, sorted_y, color='red')
        plt.xlabel('Word Count')
        plt.ylabel('Similarity')
        plt.title(method + " Similarity vs Word Count")
        plt.show()

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
    analysis_filename = filename.split(".")[0]+"_analysis.txt"
    results = parse_file(filename)
    methods = get_methods(results)
    run_analysis(methods, analysis_filename)
    #plot_word_counts(methods)

if __name__=="__main__":
    main()