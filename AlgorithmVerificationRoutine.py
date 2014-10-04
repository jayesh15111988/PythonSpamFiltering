
import Constants;

listOfIndividualProductionMessages=[];
listOfTokensForIndividualProductionMessages=[];

def getMessagesAndClassLablesWithInputFile(inputFileData,messageLabelHolder,messageHolder):

    global listOfIndividualProductionMessages;
    global listOfTokensForIndividualProductionMessages;

    #When we come here for first time, fill in data for global vectors
    if(len(listOfTokensForIndividualProductionMessages)==0):
        for individualProductionMessageToParse in inputFileData:
                tokenizedLine=individualProductionMessageToParse.split('\t');
                #First Element has to be label and second Element is Actual Message
                messageLabelHolder.append(True if tokenizedLine[0]==Constants.DEFAULT_SPAM_INDICATOR_STRING else False);
                messageHolder.append(tokenizedLine[1]);
        listOfTokensForIndividualProductionMessages=messageLabelHolder;
        listOfIndividualProductionMessages=messageHolder;
    else:
        messageLabelHolder.extend(listOfTokensForIndividualProductionMessages);
        messageHolder.extend(listOfIndividualProductionMessages);
