import random;
import pickle;
import Constants;
import csv;

def getFileData(fileName):
    spamFilteringDataFile=open(fileName,'r');
    lines = spamFilteringDataFile.readlines();
    spamFilteringDataFile.close();
    return lines;


def convertAndGetFilesDataInListFromFileWithName(fileName):
    data=[]
    realMessage=[];
    spamMessage=[];

    data=getFileData(fileName)
    for individualLine in data:
        tokenizedLine=individualLine.split('\t');
        if(len(tokenizedLine)>1):
            if(tokenizedLine[0] == Constants.DEFAULT_SPAM_INDICATOR_STRING):
                spamMessage.append(tokenizedLine[1]);
            elif (tokenizedLine[0] == Constants.DEFAULT_MESSAGE_INDICATOR_STRING):
                realMessage.append(tokenizedLine[1]);
            else:
                assert True,"Value of classification label other than spam or ham"
        else:
            #Randomize append mechanism for given message we still don't know which category it belongs to
            if(random.randint(0,1)==0):
                spamMessage.append(tokenizedLine[0]);
            else:
                realMessage.append(tokenizedLine[0]);
   
            
    #print("lengths are",len(realMessage),len(spamMessage))
    return realMessage,spamMessage;

def writeDataStructureToFileWithName(dataStructure,fileName):
    maxLengthInputSequence=len(dataStructure);
    for sequence in range(0,maxLengthInputSequence):
        pickle.dump( dataStructure[sequence], open( fileName[sequence], "wb" ) );

def getDataStructureFromFileWithName(fileName):
    try:
        return pickle.load( open( fileName, "rb" ) );
    except EOFError:
        return [];

def writeDictionaryInCSVFile(dictionaryToWrite,fileName):
    with open(fileName, 'w', newline='') as fp:
        CSVFileWriter = csv.writer(fp, delimiter=',')
        for word, frequency in dictionaryToWrite.items():
            CSVFileWriter.writerow([word, frequency]);




