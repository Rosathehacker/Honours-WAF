import pandas as pd
import json
import os
import openai
import time

 
def open_training_file():
    training_dataset_raw = pd.read_csv('trainingdata.csv')
    training_dataset_raw = training_dataset_raw.drop(training_dataset_raw.columns[0], axis=1)
    return training_dataset_raw

def open_test_file():
    test_dataset_raw = pd.read_csv('testdata.csv')
    test_dataset_raw = test_dataset_raw.drop(test_dataset_raw.columns[0], axis=1)
    return test_dataset_raw

def split_dataframe(df, chunk_size):
    chunk = []
    for i in range(0, len(df), chunk_size):
        chunk.append(df[i:i + chunk_size])
    return chunk

def check_file_ran(set_type):
    directory_path = set_type + 'ing_data_saved'
    files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
    file_count = len(files)
    return file_count

def llm_process_dataset(dataset, set_type, saved_number):
    openai.api_key = ""
    labels = ["prompt", "indicators", "type_of_injection", "keyword_signs", "command_structure_rating", "vocabulary_rating", "intent_context_rating", "semantic_risk_rating", "special_characters_rating", "repetition_rating", "language_style_rating", "language_consistency_rating", "overall_injection_score"]
    for i in range(len(dataset) - saved_number):
        thread = openai.beta.threads.create()
        index = i + saved_number
        data = dataset[index]
        dataset_processed = pd.DataFrame(columns=labels)
        for df_index, row in data.iterrows():
            print(df_index)
            response = send_to_chatgpt(row["text"], thread, labels, row["label"])
            dataset_processed = pd.concat([dataset_processed, response], ignore_index=True)
        dataset_processed.to_csv(set_type + '_data_saved\\data'+str(index)+'.csv')
def send_to_chatgpt(prompt, thread, labels, label_dataset):
    message = openai.beta.threads.messages.create(
        thread_id=thread.id,
        content=prompt,
        role="user"
        )

    run = openai.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id = "")
    while run.status == "queued" or run.status == "in_progress":
        time.sleep(1)
        run = openai.beta.threads.runs.retrieve(
            run_id=run.id, 
            thread_id=thread.id)
        if run.status == 'completed':
            break
    messages = openai.beta.threads.messages.list(thread_id=thread.id)
    last_message = messages.data[0]
    response = extract_and_convert_to_array(last_message, prompt, label_dataset)
    return response

def extract_and_convert_to_array(message: 'openai.types.beta.threads.message.Message', prompt, label):
    json_string = message.content[0].text.value
    json_data = json.loads(json_string)
    data = {
                'prompt': [prompt],
                'indicators': [json_data.get('indicators', [])],
                'type_of_injection': [json_data.get('type_of_injection', None)],
                'keyword_signs': [json_data.get('keyword_signs', [])],
                'command_structure_rating': [json_data.get('command_structure_rating', None)],
                'vocabulary_rating': [json_data.get('vocabulary_rating', None)],
                'intent_context_rating': [json_data.get('intent_context_rating', None)],
                'semantic_risk_rating': [json_data.get('semantic_risk_rating', None)],
                'special_characters_rating': [json_data.get('special_characters_rating', None)],
                'repetition_rating': [json_data.get('repetition_rating', None)],
                'language_style_rating': [json_data.get('language_style_rating', None)],
                'language_consistency_rating': [json_data.get('language_consistency_rating', None)],
                'overall_injection_score': [json_data.get('overall_injection_score', None)],
                'label': [label],
            }
    return pd.DataFrame(data)

training_dataset_raw = open_training_file()
test_dataset_raw = open_test_file()
training_data_index_ran = check_file_ran('train')
testing_data_index_ran = check_file_ran('test')
training_dataset_raw_split = split_dataframe(training_dataset_raw, 100)
test_dataset_raw_split = split_dataframe(test_dataset_raw, 100)
llm_process_dataset(training_dataset_raw_split, "training", training_data_index_ran)
llm_process_dataset(test_dataset_raw_split, "testing", testing_data_index_ran)
