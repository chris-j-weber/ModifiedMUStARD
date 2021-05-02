import csv
import json
import pandas as pd

def read_sarcasm_headlines(headlines_path):
    json_file = headlines_path
    data = []
    with open(json_file, encoding='utf8') as f:
        for line in f:
            jd = json.loads(line.encode('utf-8','replace'))
            new_row = {'is_sarcastic': int(jd['is_sarcastic']),
                       'sentence': jd['headline']}
            data.append(new_row)
    df = pd.DataFrame(data)
    return df

def read_reddit_sarcasm(reddit_train, reddit_test):
    train_csv = reddit_train
    test_csv = reddit_test
    csv_list = [train_csv, test_csv]
    df_list = []
    for csv_path in csv_list:
        data = []
        with open(csv_path, encoding='utf8') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='	')
            for row in csv_reader:
                is_sarcastic = int(row[0])
                sentence = row[1]
                context = row[9]
                new_row = {'is_sarcastic': is_sarcastic,
                           'sentence': sentence,
                           'context': context}
                data.append(new_row)
        df = pd.DataFrame(data)
        df_list.append(df)
    return df_list[0].append(df_list[1])

def read_mustard(mustard_path):
    DATA_PATH_JSON = mustard_path
    dataset_json = json.load(open(DATA_PATH_JSON))
    data = []
    for idx, ID in enumerate(dataset_json.keys()):
        new_row = {'is_sarcastic': int(dataset_json[ID]["sarcasm"]),
                    'sentence': dataset_json[ID]["utterance"],
                    'context': ' '.join(dataset_json[ID]["context"]),
                    'speaker': dataset_json[ID]["speaker"],
                    'show': dataset_json[ID]["show"],
                    'id': ID}
        data.append(new_row)
    df = pd.DataFrame(data)
    return df

def balanced_mustard_speaker_ids(df):
    # reduce to nessesary columns
    df = df[['speaker', 'is_sarcastic', 'id']]
    # Header
    final_df  = pd.DataFrame(columns = ['id', 'is_sarcastic', 'speaker'])

    for speaker in df['speaker'].unique():
        d = df[df['speaker'] == speaker]

        # slicepoint is min number that is available in either sarcastic / notsarcastic
        slicepoint = min(len(d[d['is_sarcastic'] == 1]), len(d[d['is_sarcastic'] == 0]))

        # concat rows that are sarcastic and not sarcastic
        result = pd.concat([d[d['is_sarcastic'] == 1].head(slicepoint), d[d['is_sarcastic'] == 0].head(slicepoint)])

        #concat with existing df
        final_df = pd.concat([final_df, result])                    #

    final_df.reset_index(drop=True, inplace=True)
    return list(final_df.id.values)

def balanced_dataset(df):
    speaker_ids = balanced_mustard_speaker_ids(df)
    return df[~df['id'].isin(speaker_ids)]
