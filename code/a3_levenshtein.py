import os
import numpy as np

dataDir = '/u/cs401/A3/data/'

def Levenshtein(r, h):
    """                                                                         
    Calculation of WER with Levenshtein distance.                               
                                                                                
    Works only for iterables up to 254 elements (uint8).                        
    O(nm) time ans space complexity.                                            
                                                                                
    Parameters                                                                  
    ----------                                                                  
    r : list of strings                                                                    
    h : list of strings                                                                   
                                                                                
    Returns                                                                     
    -------                                                                     
    (WER, nS, nI, nD): (float, int, int, int) WER, number of substitutions, insertions, and deletions respectively
                                                                                
    Examples                                                                    
    --------                                                                    
    >>> Levenshtein("who is there".split(), "is there".split())                         
    (0.333, 0, 0, 1)                                                                      
    >>> Levenshtein("who is there".split(), "".split())                                 
    (1.0, 0, 0, 3)                                                                           
    >>> Levenshtein("".split(), "who is there".split())                                 
    (Inf, 0, 3, 0)                                                                           
    """
    # WER, the number of substitutions, the number of insertions, and the number of deletions
    
    numReferenceWords = len(r)

    r = [""] + r
    h = [""] + h

    matrix = []
    for i in range(len(r)):
        matrix.append([])
        for j in range(len(h)):
            matrix[i].append((0, 0, 0))
    
    for i in range(1, len(r)):
        matrix[i][0] = (0, 0, i)
    
    for j in range(1, len(h)):
        matrix[0][j] = (0, j, 0)

    for i in range(1, len(r)): 
        for j in range(1, len(h)):
            matrix[i][j] = matrix[i - 1][j - 1]
            if r[i] != h[j]:
                matrix[i][j] = (matrix[i][j][0] + 1, matrix[i][j][1], matrix[i][j][2])

    if numReferenceWords:
        wer = (matrix[-1][-1][0] + matrix[-1][-1][1] + matrix[-1][-1][2])/(numReferenceWords)
    else:
        wer = float('Inf')

    return wer, matrix[-1][-1][0], matrix[-1][-1][1], matrix[-1][-1][2]

import os, fnmatch, re

def preprocess(text):
    text = text.replace(",", "")
    text = text.replace(".", "")
    text = text.replace("!", "")
    text = text.replace("?", "")
    text = text.replace("/", "")
    text = text.replace("-", "")
    text = re.sub('<.*?>', '', text)
    text = re.sub(' +', ' ', text)
    text = text.lower()
    return text

if __name__ == "__main__":

    f = open("asrDiscussion.txt", "w")

    werGoogle = []
    werKaldi = []

    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:
            # print( speaker )

            googleTranscripts = open(os.path.join( dataDir, speaker, 'transcripts.Google.txt'), 'r').readlines()
            kaldiTranscripts = open(os.path.join( dataDir, speaker, 'transcripts.Kaldi.txt'), 'r').readlines()
            transcripts = open(os.path.join( dataDir, speaker, 'transcripts.txt'), 'r').readlines()

            for i in range(len(transcripts)):
                r = preprocess(transcripts[i]).split(" ")
                hGoogle = preprocess(googleTranscripts[i]).split(" ")
                hKaldi = preprocess(kaldiTranscripts[i]).split(" ")

                wer, s, I, d = Levenshtein(r, hGoogle)
                f.write("[{speaker}] [Google] [{i}] [{wer}] S:[{s}] I:[{I}] D:[{d}]\n".format(
                    speaker=speaker, i=i, wer=wer, s=s, I=I, d=d))
                werGoogle.append(wer)

                wer, s, I, d = Levenshtein(r, hKaldi)
                f.write("[{speaker}] [Google] [{i}] [{wer}] S:[{s}] I:[{I}] D:[{d}]\n".format(
                    speaker=speaker, i=i, wer=wer, s=s, I=I, d=d))
                werKaldi.append(wer)

    print("std google:", np.std(werGoogle))
    print("mean google:", np.mean(werGoogle))
    print("std kaldi:", np.std(werKaldi))
    print("mean kaldi:", np.mean(werKaldi))

#     std google: 0.07954745253192308
# mean google: 0.9480132412512867
# std kaldi: 0.1472559785299535
# mean kaldi: 0.9262620840093712