import json
import pandas as pd
import math

files_num = 5
statistic_num = 6
representation_num = 5
foldername = "../chart_data/bomb_multi_brain/"
isMultibrain = True;

json_datas = []
json_mean_data = []

def convertIntToStatistic(value):
    if value == 1: return "cumulative_reward"
    elif value == 2: return "entropy"
    elif value == 3: return "episode_length"
    elif value == 4: return "policy_loss"
    elif value == 5: return "value_estimate"
    elif value == 6: return "value_loss"
    else: return "null"

def convertIntToRepresentation(value):
    if value == 1: return "binary_flag"
    elif value == 2: return "binary_normalized"
    elif value == 3: return "hybrid"
    elif value == 4: return "icaart"
    elif value == 5: return "zero_or_one"
    else: return "null"

def convertIntToMultiBrainNameRepresentation(value):
    if value == 1: return "Binary"
    elif value == 2: return "BinaryNormalized"
    elif value == 3: return "Hybrid"
    elif value == 4: return "ICAART"
    elif value == 5: return "ZeroOrOne"
    else: return "null"

def getFinalFilename(statisticName, representationState, representationMultibrain, sufix_num):
    if (isMultibrain):
        return foldername + representationState + "/" + statisticName + "/run_bomberman_agents_all_" + str(sufix_num) + "-0_Aprendiz" + representationMultibrain + "-tag-Info_" + statisticName + ".json"
    else:
        return foldername + representationState + "/" + statisticName + "/run_bomberman_agents_" + representationState + "_" + str(sufix_num) + "-0-tag-Info_" + statisticName + ".json"


for representation_index in range(2, representation_num + 1):
    REPRESENTATION_STATE = convertIntToRepresentation(representation_index)
    REPRESENTATION_MULTIBRAIN = convertIntToMultiBrainNameRepresentation(representation_index)

    for statistic_index in range(1, statistic_num + 1):
        STATISTIC_NAME = convertIntToStatistic(statistic_index)

        for sufix_num in range(1, files_num + 1):
            with open(getFinalFilename(STATISTIC_NAME, REPRESENTATION_STATE, REPRESENTATION_MULTIBRAIN, sufix_num)) as datafile:
                data = json.load(datafile)
                json_datas.append(data)

        # dataframe = pd.DataFrame(data)
        # print(dataframe[2].mean())
        # print(len(json_datas[0]))
        # print(json_datas[0])
        # print(len(json_datas[1]))
        # print(json_datas[1])
        # print(json_datas[1][0][2])

        maxIteration = 0
        maxIndex = 0
        for m in range(0, files_num):
            if (len(json_datas[m]) > maxIteration):
                maxIteration = len(json_datas[m])
                maxIndex = m

        # calculando a média
        for i in range(0, len(json_datas[m])):
            mean_value = 0
            for j in range(0, files_num):
                if (i < len(json_datas[j])):
                    mean_value = mean_value + json_datas[j][i][2]
                else:
                    mean_value = mean_value + json_datas[j][len(json_datas[j]) - 1][2]

            mean_value = mean_value / files_num

            line = [0, json_datas[m][i][1], mean_value]
            json_mean_data.append(line)

        print(json_mean_data)

        # calculando o desvio padrão
        for i in range(0, len(json_datas[m])):
            mean_value = json_mean_data[i][2];
            variance = 0

            for j in range(0, files_num):
                if (i < len(json_datas[j])):
                    variance = variance + ((json_datas[j][i][2] - mean_value) ** 2)
                else:
                    variance = variance + ((json_datas[j][len(json_datas[j]) - 1][2] - mean_value) ** 2)

            variance = variance / (files_num-1)
            std_deviation = math.sqrt(variance)

            json_mean_data[i][0] = std_deviation

        print(json_mean_data)

        with open(foldername + REPRESENTATION_STATE + "/" + STATISTIC_NAME + "/mean.json", "w") as outfile:
            json.dump(json_mean_data, outfile)

        # limpando lista
        json_datas.clear()
        json_mean_data.clear()