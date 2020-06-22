import csv
import ast
import json
import pandas as pd
import numpy as np
from scipy import stats
from tqdm import tqdm
from itertools import permutations


def idomaar_to_csv(infile, outfile, mode):
    with open(infile, "r") as fr, open(outfile,'w', newline='') as fw:
        if mode == "persons": csv_fields = ("id", "name")
        elif mode == "tracks": csv_fields = ("id", "name", "playcount", "artist")
        elif mode == "sessions": csv_fields = ("id", "session", "len") 
        elif mode == "playlists": csv_fields = ("id", "playlist", "len")
        writer = csv.DictWriter(fw, csv_fields, delimiter=',')
        writer.writeheader()

        record = {}
        for line in fr:
            fields = line.strip().split("\t")
            
            record[csv_fields[0]] = fields[1]
            if mode == "persons":
                description = json.loads(fields[3])
                if "&" not in description["name"]:
                    record[csv_fields[1]] = description["name"]
                else:
                    continue
            elif mode == "tracks":
                description = json.loads(fields[3])
                artist, name = description["name"].strip().split("/_/")
                record[csv_fields[1]] = name
                record[csv_fields[2]] = description["playcount"]
                record[csv_fields[3]] = artist
            elif mode == "sessions" or mode == "playlists":
                if mode == "sessions":
                    description = json.loads(fields[3].split()[1])['objects']
                else:
                    description = json.loads(fields[4])['objects']
                tracks = []
                if len(description) < 2:
                    continue
                for track in description:
                    tracks.append(track['id'])
                record[csv_fields[1]] = tracks
                record[csv_fields[2]] = len(tracks)
            
            writer.writerow(record)
            
def build_dataset(data, tracks, artist_index, filepath, mode='session'):
    with open(filepath,'w', newline='') as fw:
        csv_fields = ('first', 'second')
        writer = csv.DictWriter(fw, csv_fields, delimiter=',')
        writer.writeheader()
        for _, row in data.iterrows():
            song_indx = ast.literal_eval(row[mode])
            art = []
            record = {}

            for song in song_indx:
                track = tracks[tracks.id == song]
                if not track.empty:
                    artist = track.iloc[0]['artist']
                    if artist in artist_index:
                        art.append(artist_index[artist])
            pairs = permutations(set(art), 2)
            for pair in pairs:
                record['first'] = pair[0]
                record['second'] = pair[1]
                writer.writerow(record)
                
def find_similar(artist, model, index_artist, artist_index, n=10):
    emb_layer = model.get_layer('artist_embedding')
    emb_weights = emb_layer.get_weights()[0]
    emb_weights = emb_weights / np.linalg.norm(emb_weights, axis = 1).reshape((-1, 1))
    try:
        dists = np.dot(emb_weights, emb_weights[artist_index[artist]])
    except KeyError:
        print(f'{name} Not Found.')
        return
    
    idxs = np.argsort(dists)[-n-1:-1]
    print(f'{n} artists most similar to {artist}')
    for c in reversed(idxs):
        print(f'Artist: {index_artist[c]} | Distance: {dists[c]:.2}')
        
def stat_eval(model, test_data):
    p_values = []
    for i in range(100):
        losses = []
        for first, second in test_data[i:i+100, :]:
            data_x = {}
            data_x['first'], data_x['second'] = np.array([first]), np.array([second])
            data_y = np.array([1])
            losses.append(model.evaluate(data_x, data_y, batch_size=None, steps=1, verbose=0))
        losses = np.array(losses)
        t, p = stats.ttest_1samp(losses, 1)
        p_values.append(p)

    return np.array(p_values).mean()
              