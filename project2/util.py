import numpy as np

tags = []
tags.append("START")
tags.append("STOP")
tags.append("COMMA")
tags.append("PERIOD")
tags.append("QUESTION_MARK")
tags.append("EXCLAMATION_POINT")
tags.append("COLON")
tags.append("SPACE")

def process_line(line, islabels):
    line = line.strip()
    sample = line.split(' ')

    if islabels:
        sample[0:0] = [tags[0]]
        sample.extend([tags[1]])
    else:
        sample[0:0] = [""]
        sample.extend([""])
    
    return sample


def read_file(filename):
    """Read a data file and return data matrix and labels"""
    
    #read data file
    f = open("punctuationDataset/{}Sentences.txt".format(filename))
    lines = f.read().split('\n')
    data = (list(process_line(i, False) for i in lines if len(i.strip()) > 0))
    f.close()
    
    #read labels file
    f = open("punctuationDataset/{}Labels.txt".format(filename))
    lines = f.read().split('\n')
    labels = (list(process_line(i, True) for i in lines if len(i.strip()) > 0))
    f.close()
    
    return data, labels

if __name__ == "__main__":
    print np.array([""])
    data, labels = read_file('training')