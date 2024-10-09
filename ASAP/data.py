from scipy.stats import spearmanr

import random
import sys

import matplotlib.pyplot as plt
import seaborn as sns

asap_ranges = {
    1: (2.0, 12.0),
    2: (1.0, 6.0),
    3: (0.0, 3.0),
    4: (0.0, 3.0),
    5: (0.0, 4.0),
    6: (0.0, 4.0),
    7: (0.0, 30.0),
    8: (0.0, 60.0),
    9: (0.5, 9.0),
    10: (1.0, 24.0),
}


asap_essay_lengths = {
    1: 649,
    2: 704,
    3: 219,
    4: 203,
    5: 258,
    6: 289,
    7: 371,
    8: 1077,
    9: 415,
    10: 1024,
    11: 252
}


def fix_score(score, prompt):
    """
    fix the predicted score
    """
    # if prompt == 9:  # telis
    #     int_part = float(int(score))
    #     float_part = score - int_part
    #     result = int_part
    #     if float_part < 0.25:
    #         result = int_part
    #     elif float_part < 0.75:
    #         result = int_part + 0.5
    #     else:
    #         result = int_part + 1

    #     min_score, max_score = asap_ranges[prompt]
    #     if result < min_score:
    #         return min_score
    #     elif result > max_score:
    #         return max_score
    #     else:
    #         return result

    # elif prompt <= 10:
    #     min_score, max_score = asap_ranges[prompt]
    #     if score < min_score:
    #         return min_score
    #     elif score > max_score:
    #         return max_score
    #     else:
    #         return round(score)
    # else:
    #     return score

    min_score, max_score = asap_ranges[prompt]

    norm_score = (score / (60 / max_score)) + min_score

    print(norm_score, min_score, max_score, score)

    return norm_score

    # return score


def is_zh(s):
    # '包含汉字的返回TRUE'
    for c in s:
        if c >= '\u4e00' and c <= '\u9fa5':
            return True
    return False


def load_asap_data(data_file, max_len=1024, data_sample_rate=1.0):
    ids = []
    texts = []
    labels = []
    sample_index = 0
    with open(data_file) as fin:
        for line in fin:
            rand_value = random.random()
            if rand_value > data_sample_rate:
                continue
            line = line.strip()
            line_vec = line.split("\t")
            if len(line_vec) == 3:
                ids.append(line_vec[0])
                if len(line_vec[1].split(" ")) >= max_len:
                    line_vec[1] = " ".join(line_vec[1].split(" ")[0:max_len])
                texts.append(line_vec[1])
                labels.append(float(line_vec[2]))
            else:
                ids.append(str(sample_index))
                sample_index += 1
                if is_zh(line_vec[0]) and len(line_vec[0].replace(" ", "")) >= max_len:
                    line_vec[0] = line_vec[0].replace(" ", "")[0:max_len]
                elif len(line_vec[0].split(" ")) >= max_len:
                    line_vec[0] = " ".join(line_vec[0].split(" ")[0:max_len])
                texts.append(line_vec[0])
                labels.append(float(line_vec[1]))
    for id, text, label in zip(ids, texts, labels):
        yield (id, text, label)


if __name__ == "__main__":

    file = str(sys.argv[1])

    # TODO: CHANGE THIS WHENEVER MAKING A NEW FIX
    # Perform Spearman's Correlation
    with open(f'results/fix1/pred-crc-{file}.txt', 'r') as src:
        y, X = list(zip(*[(float(line.split()[0]), int(float(line.split()[1]))) for line in src]))

    # TODO: CHANGE THIS WHENEVER MAKING A NEW FIX
    with open(f'results/fix2-plots-corr/spearman-{file}.txt', 'w+') as dst:
        dst.write("Spearman correlation on the transformed outputs and the gold labels\n")
        dst.write(f"{str(spearmanr(X, y))}\n")

    sns.scatterplot(x=X, y=y)
    plt.xlabel("Model Output")
    plt.ylabel(f"Prompt {file} Gold Labels")
    plt.title(f"Model Output vs. Prompt {file} Gold Labels")

    # TODO: CHANGE THIS WHENEVER MAKING A NEW FIX
    plt.savefig(f'results/fix2-plots-corr/spearman-{file}.png')
