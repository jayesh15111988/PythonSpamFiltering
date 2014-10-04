
import Constants;

def getMessagesAndClassLablesWithInputFile(inputFileData,messageLabelholder,messageHolder):
    for individualProductionMessageToParse in inputFileData:
            tokenizedLine=individualProductionMessageToParse.split('\t');
            #First Element has to be label and second Element is Actual Message
            messageLabelholder.append(True if tokenizedLine[0]==Constants.DEFAULT_SPAM_INDICATOR_STRING else False);
            messageHolder.append(tokenizedLine[1]);