import pandas as pd
import sys
import os
import copy

class CSV_Files:

    @staticmethod
    def get_content_of_csv_files(path: "str", files_list: "list", drop_culm:list[str]=[]):
        content_list = []
        for item in files_list:
            content_list.append(pd.read_csv(str(path + item)))
            content_list[-1] = content_list[-1].drop(drop_culm, axis=1)
            try:
                content_list[-1] = content_list[-1].drop('weight', axis=1)
            except:
                pass
            null_values = content_list[-1].isnull().sum()
            null_row_cunt = null_values.sum()
            if null_row_cunt > 0:
                print(item, null_row_cunt)
                for j, jtem in enumerate(null_values):
                    if jtem > 0 :
                        print(list(content_list[-1])[j], jtem)
            content_list[-1] = content_list[-1].dropna()
        return content_list

    @staticmethod
    def scale_data(data, drop_centrality):
        scale_vector = [10000, 1000000, 1000000000, 100000, 1000000000, 1000000000, 1000000, 100000, 10000, 1, 1, 1, 10, 10, 100, 1, 100]
        scaled_data = copy.deepcopy(data)
        features_list = list(data.keys())
        j = 0
        for i, item in enumerate(features_list):
            if not(item in drop_centrality):
                scaled_data[item] = data[item]/scale_vector[j]
                j += 1
        data = copy.deepcopy(scaled_data)
        del scaled_data
        return data

    pass