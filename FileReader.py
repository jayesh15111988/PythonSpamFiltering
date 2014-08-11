import random;

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
            if(tokenizedLine[0] == "spam"):
                spamMessage.append(tokenizedLine[1]);
            elif (tokenizedLine[0] == "ham"):
                realMessage.append(tokenizedLine[1]);
            else:
                assert (true),"Value of classification label other than spam or ham"
        else:
            #Randomize append mechanism for given message we still don't know which category it belongs to
            if(random.randint(0,1)==0):
                spamMessage.append(tokenizedLine[0]);
            else:
                realMessage.append(tokenizedLine[0]);
   
            
    #print("lengths are",len(realMessage),len(spamMessage))
    return realMessage,spamMessage;



