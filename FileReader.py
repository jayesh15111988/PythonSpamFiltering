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
        if(tokenizedLine[0] == "spam"):
            spamMessage.append(tokenizedLine[1]);
        elif (tokenizedLine[0] == "ham"):
            realMessage.append(tokenizedLine[1]);
        else:
            assert (true),"Value of classification label other than spam or ham"
    #print("lengths are",len(realMessage),len(spamMessage))
    return realMessage,spamMessage;



