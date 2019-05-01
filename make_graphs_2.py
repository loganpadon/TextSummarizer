import sys

def parse_file(filename):
    with open(filename, "r", encoding="utf8") as file_lines:
        results = []
        entry = {}
        for line in file_lines:
            if line[0] == "-":
                if entry:
                    results.append(entry)
            else:
                key, value = line.split(":")
                if key != "Word count" and len(key.split()) > 2:
                    method, metric = key.split()
                    if method not in entry.keys():
                        entry[method] = {}
                    entry[method][metric] = value
                else:
                    entry[key] = value
        return results
            

def main():
    if len(sys.argv) != 2:
        print("Usage: python make_graphs.py [filename]")
        exit()
    filename = sys.argv[1]
    results = parse_file(filename)

if __name__=="__main__":
    main()