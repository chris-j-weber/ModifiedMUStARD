import json
import pandas as pd
import matplotlib.pyplot as plt


with open('res.json', 'r', encoding='utf8') as json_file:
    x = json.load(json_file)

ypred = x['pred']
ytrue = x['true']

mustard_path = "data/sarcasm_data.json"

def read_mustard_identity():
    DATA_PATH_JSON = mustard_path
    dataset_json = json.load(open(DATA_PATH_JSON))
    data = []
    for idx, ID in enumerate(dataset_json.keys()):
        if dataset_json[ID]['show'] == 'FRIENDS':
            data.append(dataset_json[ID]["speaker"])
    return data
speaker = read_mustard_identity()

dataset = {'speaker': speaker, 'y':ytrue, 'pred':ypred}
print(dataset)

df = pd.DataFrame(dataset)
#print(df)


speakerlist = set(dataset['speaker'])
print(speakerlist)

new_df = {}
for s in speakerlist:
    n = df[df['speaker']==s]
    richtig = n[n['y']==n['pred']]
    y1 = len(n[n['y'] == 1])
    pred1 = len( n[n['pred']==1])
    new_df[s] = [len(n), len(richtig), y1, pred1]

print(new_df)
new_df = pd.DataFrame(new_df, index=["insg.", "korrekt", 'y=1', 'pred=1'])
new_df = new_df.transpose().apply(lambda x: x, axis=1).plot(kind="bar", stacked=False)
new_df.plot(kind="bar")
plt.title("Prediction by Speaker")
plt.xlabel("speakerr")
plt.ylabel("#")
plt.show()


