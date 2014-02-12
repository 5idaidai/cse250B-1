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
    sentences = "punctuationDataset/{}Sentences.txt".format(filename)
    print sentences
    f = open(sentences)
    lines = f.read().split('\n')
    data = (list(process_line(i, False) for i in lines if len(i.strip()) > 0))
    f.close()
    
    #read labels file
    labelsfile = "punctuationDataset/{}Labels.txt".format(filename)
    print labelsfile
    f = open(labelsfile)
    lines = f.read().split('\n')
    labels = (list(process_line(i, True) for i in lines if len(i.strip()) > 0))
    f.close()
    
    return data, labels

if __name__ == "__main__":
    print np.array([""])
    data, labels = read_file('training')