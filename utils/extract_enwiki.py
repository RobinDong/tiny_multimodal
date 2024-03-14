import glob

from multiprocessing import Pool

n_jobs = 12
input_path = "/mnt/backup_3080ti/enwiki/text/*/*"
output_path = "enwiki/"
min_words = 128


# remove "&lt;templateXXXXXXX&gt;"
def line_process(line):
    length = len(line)
    ans = ""
    read = True
    index = 0
    while index < length:
        if line[index : index + 4] == "&lt;":
            read = False
            index += 4
        elif line[index : index + 4] == "&gt;":
            read = True
            index += 4
        elif read:
            ans += line[index]
        index += 1
    return ans


def article_process(lines):
    article = []
    for line in lines:
        if line.startswith("<doc id=") or line.startswith("</doc>"):
            continue
        line = line_process(line)
        if len(line.split(" ")) >= min_words:
            article += line
    return article


def task(pair):
    job_id, file_list = pair
    for filename in file_list:
        with open(filename, "r", encoding="utf-8") as fp:
            lines = article_process(fp.readlines())
        with open(output_path + str(job_id) + ".txt", "a", encoding="utf-8") as fp:
            for line in lines:
                fp.write(line)


def extract():
    file_list = glob.glob(input_path)
    width = len(file_list) // n_jobs
    lists = []
    for index in range(n_jobs):
        lists.append((index, file_list[index * width : (index + 1) * width]))
    with Pool(n_jobs) as pool:
        pool.map(task, lists)


if __name__ == "__main__":
    extract()
