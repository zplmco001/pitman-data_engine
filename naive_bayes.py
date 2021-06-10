from math import exp, pi, sqrt


def get_stats(numbers):
    mean = sum(numbers) / float(len(numbers))
    variance = sum([(x - mean) ** 2 for x in numbers]) / float(len(numbers) - 1)
    return mean, variance


def gaussian_prob(x, mean, variance):
    exponent = exp(-((x - mean) ** 2 / (2 * variance)))
    return (1 / (sqrt(2 * pi * variance))) * exponent


def divide_to_classes(data):
    divided = dict()
    for i in range(len(data)):
        row = data[i]
        class_value = row[-1]
        if class_value not in divided:
            divided[class_value] = list()
        divided[class_value].append(row)
    return divided


def summary(data):
    divided = divide_to_classes(data)
    summary_result = dict()
    for class_value, rows in divided.items():
        summaries = [(get_stats(column), len(column)) for column in zip(*rows)]
        del (summaries[-1])
        summary_result[class_value] = summaries
    return summary_result


def calculate(summaries, row):
    total_rows = sum([summaries[label][0][1] for label in summaries])
    probabilities = dict()
    for class_value, class_summaries in summaries.items():
        probabilities[class_value] = summaries[class_value][0][1] / float(total_rows)
        for i in range(len(class_summaries)):
            mean, variance = class_summaries[i][0]
            probabilities[class_value] *= gaussian_prob(row[i], mean, variance)
    return probabilities


'''dataset = [[3.393533211, 2.331273381, 0],
           [3.110073483, 1.781539638, 0],
           [1.343808831, 3.368360954, 0],
           [3.582294042, 4.67917911, 0],
           [2.280362439, 2.866990263, 0],
           [7.423436942, 4.696522875, 1],
           [5.745051997, 3.533989803, 1],
           [9.172168622, 2.511101045, 1],
           [7.792783481, 3.424088941, 1],
           [7.939820817, 0.791637231, 1]]

summaries = summary(dataset)
probabilities = calculate(summaries, dataset[6])
print(probabilities)'''
