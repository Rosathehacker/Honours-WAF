import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import f1_score, precision_score, recall_score


import os
import ast

def openfiles(set_type):
    directory_path = set_type + 'ing_data_saved'
    labels = ["prompt", "indicators", "type_of_injection", "keyword_signs", "command_structure_rating", "vocabulary_rating", "intent_context_rating", "semantic_risk_rating", "special_characters_rating", "repetition_rating", "language_style_rating", "language_consistency_rating", "overall_injection_score"]
    dataset = pd.DataFrame(columns=labels)
    for file in [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]:
        df = pd.read_csv(set_type + 'ing_data_saved\\'+str(file))
        dataset = pd.concat([dataset, df], ignore_index=True)
    dataset = dataset.drop(dataset.columns[13], axis=1)
    return dataset

def open_training_file():
    training_dataset_raw = pd.read_csv('trainingdata.csv')
    training_dataset_raw = training_dataset_raw.drop(training_dataset_raw.columns[0], axis=1)
    return training_dataset_raw

def open_test_file():
    test_dataset_raw = pd.read_csv('testdata.csv')
    test_dataset_raw = test_dataset_raw.drop(test_dataset_raw.columns[0], axis=1)
    return test_dataset_raw

def check_llm_accuracy_dataset_pre_mla(techniques, dataset):
    positive_negative_array = []
    false_true_array = []
    for df_index, row in dataset.iterrows():
        for i in range(len(techniques)):
            if row["type_of_injection"]== techniques[i]:
                positive_negative_array.append(1)
                break
            elif row["type_of_injection"] == "benign":
                 positive_negative_array.append(0)
                 break
            else:
                continue
    for df_index, row in dataset.iterrows():
        
        if dataset["label"].iloc[df_index] == positive_negative_array[df_index]:

            false_true_array.append(1)
        else:

            false_true_array.append(0)
    f1, f2 = calculate_f1_f2(positive_negative_array, training_dataset["label"]) 
    return f1, f2

def calculate_f1_f2(predictions, labels):
    binary_predictions = [1 if p >= 0.5 else 0 for p in predictions]
    f1 = f1_score(labels, binary_predictions, average='weighted')
    precision = precision_score(labels, binary_predictions, average='weighted')
    recall = recall_score(labels, binary_predictions, average='weighted')
    f2 = (1 + 2**2) * (precision * recall) / ((2**2 * precision) + recall)
    return f1, f2

def convert_to_numberical(dataset, indicators, techniques, columns_labels):
    dataset_numerised = pd.DataFrame(columns=columns_labels)
    indicators_list = dataset['indicators']
    indicators_list = indicators_list.tolist()
    for i in range(len(indicators_list)):
        indicators_list[i] = ast.literal_eval(indicators_list[i])
    keyword_list = dataset['keyword_signs']
    keyword_list = keyword_list.tolist()
    no_of_keywords = []
    for i in range(len(keyword_list)):
        keyword_list[i] = ast.literal_eval(keyword_list[i])
        no_of_keywords.append(len(keyword_list[i]))
    indicators_features = np.zeros((len(indicators_list), len(indicators)))
    for index, row in enumerate(indicators_list):
        for colidx, indicator in enumerate(indicators):
            value = row.count((indicator))
            indicators_features[index, colidx] = value
    techniques_features = np.zeros((len(dataset), len(techniques)))
    for index, row in dataset.iterrows():
        for colidx, technique in enumerate((techniques)):
            value = row["type_of_injection"].count(technique)
            techniques_features[index, colidx] = value
    positive_negative_array = []
    for df_index, row in dataset.iterrows():
        for i in range(len(techniques)):
            if row["type_of_injection"]== techniques[i]:
                positive_negative_array.append(1)
                break
            elif row["type_of_injection"] == "benign":
                 positive_negative_array.append(0)
                 break
            else:
                continue
    for df_index, row in dataset.iterrows():
        data = {
            "is_injection": [float(positive_negative_array[df_index])],
            "Instructional overrides": [indicators_features[df_index, 0]],
            "Role reversal/impersonation": [indicators_features[df_index, 1]],
            "System Information requests": [indicators_features[df_index, 2]],
            "Obfuscation/Encoding": [indicators_features[df_index, 3]], 
            "Contextual Confusion": [indicators_features[df_index, 4]], 
            "Out of scope requests": [indicators_features[df_index, 5]], 
            "Nested instructions": [indicators_features[df_index, 6]], 
            "Cross domain prompt injection": [indicators_features[df_index, 7]], 
            "Character limitations/manipulation": [indicators_features[df_index, 8]], 
            "Prompt chaining/Prompt engineering": [indicators_features[df_index, 9]], 
            "Data exfiltration attempts": [indicators_features[df_index, 10]], 
            "Timing attacks/Rate limiting manipulation": [indicators_features[df_index, 11]], 
            "Special Character Injection": [indicators_features[df_index, 12]], 
            "Recursion/Looping": [indicators_features[df_index, 13]], 
            "Model Parameter manipulation": [indicators_features[df_index, 14]], 
            "Direct Injection": [techniques_features[df_index, 0]], 
            "Indirect Injection": [techniques_features[df_index, 1]], 
            "Cognitive Hacking": [techniques_features[df_index, 2]], 
            "Repetition": [techniques_features[df_index, 3]], 
            "Syntactical transformation": [techniques_features[df_index, 4]], 
            "Few-Shot": [techniques_features[df_index, 5]], 
            "Text Completion": [techniques_features[df_index, 6]], 
            "Prompt Leakage": [techniques_features[df_index, 7]], 
            "Token Smuggling": [techniques_features[df_index, 8]], 
            "Adversarial Examples": [techniques_features[df_index, 9]], 
            "In-Context Learning Exploitation": [techniques_features[df_index, 10]], 
            "Model Extraction": [techniques_features[df_index, 11]], 
            "Factuality Attacks": [techniques_features[df_index, 12]], 
            "Benign": [techniques_features[df_index, 13]],
            "keywords_number": [float(no_of_keywords[df_index])], 
            "command_structure_rating": [float(row["command_structure_rating"])], 
            "vocabulary_rating": [float(row["vocabulary_rating"])], 
            "intent_context_rating": [float(row["intent_context_rating"])], 
            "semantic_risk_rating": [float(row["semantic_risk_rating"])], 
            "special_characters_rating": [float(row["special_characters_rating"])], 
            "repetition_rating": [float(row["repetition_rating"])], 
            "language_style_rating": [float(row["language_style_rating"])], 
            "language_consistency_rating": [float(row["language_consistency_rating"])], 
            "overall_injection_score": [float(row["overall_injection_score"])]}
        data_numerised_df = pd.DataFrame(data)
        dataset_numerised = pd.concat((dataset_numerised, data_numerised_df), ignore_index=True)
    return dataset_numerised

def mla(training_dataset_numerised, training_dataset, columns_labels):
    xgb_train = xgb.DMatrix(training_dataset_numerised[columns_labels], label=training_dataset["label"])
    params = {
  "colsample_bynode": 0.8,
  "learning_rate": 1,
  "max_depth": 5,
  "num_parallel_tree": 100,
  "objective": "binary:logistic",
  "subsample": 0.8,
  "tree_method": "hist",
  "device": "cuda",
}
    num_rounds = 100
    model = xgb.train(params, xgb_train, num_rounds)
    return model

def test_mla(testing_dataset_numerised, testing_dataset, mla_model, columns_labels):
    xgb_test = xgb.DMatrix(testing_dataset_numerised[columns_labels])
    predictions = mla_model.predict(xgb_test)
    f1, f2 = calculate_f1_f2(predictions, testing_dataset["label"])
    return f1, f2

indicators = ["Instructional overrides",
              "Role reversal/impersonation",
              "System Information requests",
              "Obfuscation/Encoding",
              "Contextual Confusion",
              "Out of scope requests",
              "Nested instructions", 
              "Cross domain prompt injection",
              "Character limitations/manipulation",
              "Prompt chaining/Prompt engineering",
              "Data exfiltration attempts",
              "Timing attacks/Rate limiting manipulation",
              "Special Character Injection",
              "Recursion/Looping",
              "Model Parameter manipulation"]
techniques = ["Direct Injection",
              "Indirect Injection",
              "Cognitive Hacking",
              "Repetition", 
              "Syntactical transformation",
              "Few-Shot", "Text Completion",
              "Prompt Leakage",
              "Token Smuggling",
              "Adversarial Examples",
              "In-Context Learning Exploitation",
              "Model Extraction",
              "Factuality Attacks",
              "benign"]
columns_labels = [
    "is_injection",
    "Instructional overrides",
    "Role reversal/impersonation",
    "System Information requests",
    "Obfuscation/Encoding", 
    "Contextual Confusion", 
    "Out of scope requests", 
    "Nested instructions", 
    "Cross domain prompt injection", 
    "Character limitations/manipulation", 
    "Prompt chaining/Prompt engineering", 
    "Data exfiltration attempts", 
    "Timing attacks/Rate limiting manipulation", 
    "Special Character Injection", 
    "Recursion/Looping", 
    "Model Parameter manipulation", 
    "Direct Injection", 
    "Indirect Injection", 
    "Cognitive Hacking", 
    "Repetition", 
    "Syntactical transformation", 
    "Few-Shot", 
    "Text Completion", 
    "Prompt Leakage", 
    "Token Smuggling", 
    "Adversarial Examples", 
    "In-Context Learning Exploitation", 
    "Model Extraction", 
    "Factuality Attacks",
    "Benign",
    "keywords_number", 
    "command_structure_rating", 
    "vocabulary_rating", 
    "intent_context_rating", 
    "semantic_risk_rating", 
    "special_characters_rating", 
    "repetition_rating", 
    "language_style_rating", 
    "language_consistency_rating", 
    "overall_injection_score"]
training_dataset = openfiles("train")
testing_dataset = openfiles("test")
training_ac, training_f1 ,training_f2 = check_llm_accuracy_dataset_pre_mla(techniques, training_dataset)
testing_ac, testing_f1, testing_f2 = check_llm_accuracy_dataset_pre_mla(techniques, testing_dataset)
training_dataset_numerised = convert_to_numberical(training_dataset, indicators, techniques, columns_labels)
testing_dataset_numerised = convert_to_numberical(testing_dataset, indicators, techniques, columns_labels)
mla_model = mla(training_dataset_numerised, training_dataset, columns_labels)
f1_mla, f2_mla = test_mla(testing_dataset_numerised, testing_dataset, mla_model, columns_labels)


 
