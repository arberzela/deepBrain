from PatientTranscripts import *

def samples_per_words(percentages, sentences, samples):
    """
    Maps the number of samples with the corresponding words in a sentence.

    :type percentages: list of floats
                       how much percent a word occupies in the sentence
    :type sentences: list of strings
    :type samples: ndarray with shape (#samples, #channels)
                   contains the samples for the given sentence
    :return: dictionary with
             word strings as keys
             ndarray with the samples for each word as corresponding values
    """
    labeledData = list()
    
    for i, sentence in enumerate(sentences):
        sentence_dict = dict()
        last_index = 0
        for j in range(len(sentence)):
            index = np.int(round(percentages[i][j] * samples[i].shape[0]))
            sentence_dict[sentence[j]] = samples[i][last_index:index + last_index, :]
            last_index += index

            # some samples left out due to rounding error
            # assert(last_index == samples[i].shape[0])
            
        labeledData.append(sentence_dict)

    return labeledData


def AlignData(patient, align_word_data = False):
    
    aligned_data = list()
    transcripts = get_all_transcripts()[patient - 1].transcripts

    os.chdir('C:\\Users\\user\\Desktop\\Master Project\\ongoing')
    with open('patient_' + str(patient) + '.pickle', 'rb') as f:
        data = cPickle.load(f)
    assert(len(transcripts) == len(data))
    
    if align_word_data: #align each word with the brain data   
        for day in range(1, len(transcripts) + 1):
            labeledData = samples_per_words(transcripts[day].word_percentages(), transcripts[day].sentences, data[day])
            aligned_data += labeledData
    else:   #align each sentence with the brain data
        for day in range(1, len(transcripts) + 1):
            for sentence in range(len(transcripts[day].sentences)):
                #now we represent the sentences as strings with the space delimiter, with all uppercase letters
                labeledData = (data[day][sentence], ' '.join(transcripts[day].sentences[sentence]).upper())
                aligned_data.append(labeledData)

    return aligned_data

def saveAlignedData(patient = None, align_word_data = False):

    if patient is None:
        for patientNr in range(1, 5):
            aligned_data = AlignData(patientNr)
            with open('patient' + str(patientNr) + 'Aligned.pickle', 'wb') as f:
                cPickle.dump(aligned_data, f)
    elif patient in [1, 2, 3, 4]:
        aligned_data = AlignData(patient)
        with open('patient' + str(patient) + 'Aligned.pickle', 'wb') as f:
            cPickle.dump(aligned_data, f)
    else:
        raise ValueError('No patient with this number!!')

def loadAlignedData(patient):

    if patient in [1, 2, 3, 4]:
        with open('patient' + str(patient) + 'Aligned.pickle', 'rb') as f:
            alignedData = cPickle.load(f)
        return alignedData
    else:
        raise ValueError('No patient with this number!!')
