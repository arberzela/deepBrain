import argparse
import json
import os, re
import shutil
from pathlib import Path
from six.moves import cPickle


def write_json(pkl_file, output_file, lower_no_ummlaut=False):
    '''
    Create JSON-line description file from a single .pickle file.
    '''
    labels = []
    durations = []
    features = []
    with open(pkl_file, 'rb') as file:
        data = cPickle.load(file)
    for example in data:
        if lower_no_ummlaut:
            label = re.sub('ä', 'ae', example[1].lower())
            label = re.sub('ü', 'ue', label)
            label = re.sub('ö', 'oe', label)
        else:
            label = example[1]
        feature = example[0]
        duration = feature.shape[0]
        labels.append(label)
        features.append(feature.tolist())
        durations.append(duration)
    print('Processed '+str(len(labels))+' utterances!')
    print('Number of channels', feature.shape[1])
    print('Min duration: ', min(durations))
    print('Max duration: ', max(durations), '\n')
    
    with open(output_file, 'w') as out_file:
        for i in range(len(labels)):
            line = json.dumps({'feature': features[i], 'duration': durations[i],
                               'text': labels[i]})
            out_file.write(line + '\n')


def main(data_dir, patientNr, lower_no_ummlaut):
    try:
        os.chdir(data_dir)
        write_dir = os.path.join('.', 'json', 'Patient' + str(patientNr))
        if os.path.isdir(write_dir):
            shutil.rmtree(write_dir)
        path = Path(write_dir)
        path.mkdir(parents=True, exist_ok=True)
        
        print('Creating JSON files for patient ' + str(patientNr) + '!')
        for partition in ['train', 'valid', 'test']:
            print('Processing ' + partition + 'set!')
            if partition == 'train':
                out_file = os.path.join(write_dir, partition + '.json')
                pkl_file = os.path.join(data_dir, partition,
                                        'train' + str(patientNr) + '.pickle')
                write_json(pkl_file, out_file, lower_no_ummlaut)

            if partition == 'valid':
                out_file = os.path.join(write_dir, partition + '.json')
                pkl_file = os.path.join(data_dir, partition,
                                        'valid' + str(patientNr) + '.pickle')
                write_json(pkl_file, out_file, lower_no_ummlaut)

            if partition == 'test':
                out_file = os.path.join(write_dir, partition + '.json')
                pkl_file = os.path.join(data_dir, partition,
                                        'test' + str(patientNr) + '.pickle')
                write_json(pkl_file, out_file, lower_no_ummlaut)
        print('\n')
                
    except FileNotFoundError:
        print('The data directory specified does not exist!\n')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default=os.path.join('..', 'data'),
                        help='Path to data directory')
    parser.add_argument('--lower_no_ummlaut', type=bool,
                        default=False,
                        help='labels lowercase and with no ummlaut characters.')
    args = parser.parse_args()
    for patientNr in range(1, 5):
        main(args.data_dir, patientNr, args.lower_no_ummlaut)
