from FileReader import *
from StringUtilities import *
import re

def getAndFilterMessagesInDataStructureWithFileName(inputFileName):
    separatedMessageData= convertAndGetFilesDataInListFromFileWithName(inputFileName);
    dynamicAttributrMappingDictionary={};
    dynamicAttributrMappingDictionary['numberOfCapitalLettersInMessage']=1;
    dynamicAttributrMappingDictionary['lengthOfLongestAllCapitalword']=2;
    dynamicAttributrMappingDictionary['messageTypeIndicator']=3;

    fullMessageDetailsWithRegularAndSpamMessage=separatedMessageData[0]+separatedMessageData[1];

    

    #print(len(fullMessageDetailsWithRegularAndSpamMessage));
    filteredMessagesWithPunctuationElimination=[];

    collectionOfVectorsOfAllStatements=[];

    frequencyOfWordsInRegularMessages={};
    frequencyOfWordsInSpamMessages={};

    for i,individualMessage in enumerate(fullMessageDetailsWithRegularAndSpamMessage):
        filteredMessagesWithPunctuationElimination.append(getStringWithPunctuationsRemoved(individualMessage));


    for externalIndex,individualPuncutationRemovedMessage in enumerate(filteredMessagesWithPunctuationElimination):
        tokenizedDataIntoArray=list(filter(None, re.split('\W+',individualPuncutationRemovedMessage)));
        vectorForGivenMessage=[-1]*len(dynamicAttributrMappingDictionary);
        
        #initialize these variable for each message that will be encountered
        
        lengthOfLongestAllCapitalword=-1;

        for index,individualAttributeTokens in enumerate(tokenizedDataIntoArray):
            lengthOfCurrentSelectedToken=len(individualAttributeTokens);
            if(individualAttributeTokens.isupper() and lengthOfCurrentSelectedToken>lengthOfLongestAllCapitalword):
                vectorForGivenMessage[1]=len(individualAttributeTokens);
                lengthOfLongestAllCapitalword=lengthOfCurrentSelectedToken;

            
            if not(individualAttributeTokens in dynamicAttributrMappingDictionary):
                dynamicAttributrMappingDictionary[individualAttributeTokens]=len(dynamicAttributrMappingDictionary)+1
                vectorForGivenMessage=vectorForGivenMessage+[-1];
            indexForGivenVector = dynamicAttributrMappingDictionary[individualAttributeTokens]-1;
        
            if (vectorForGivenMessage[indexForGivenVector]==-1):
                vectorForGivenMessage[indexForGivenVector]=0;
            else:
                vectorForGivenMessage[indexForGivenVector]=vectorForGivenMessage[indexForGivenVector]+1;

            if(externalIndex<len(separatedMessageData[0])):
                #We have actual message
                vectorForGivenMessage[2]='m';
                storeInProbabilityStoreForValue(frequencyOfWordsInRegularMessages,individualAttributeTokens)
            else:
                #We have spam as we already crossed message boundary
                vectorForGivenMessage[2]='s';
                storeInProbabilityStoreForValue(frequencyOfWordsInSpamMessages,individualAttributeTokens)
         #Here end tokenizer loop and we go get next message - Might it be a spam or not   
        vectorForGivenMessage[0]=getNumberOfCapitalLettersFromWord(individualPuncutationRemovedMessage);
        collectionOfVectorsOfAllStatements.append(vectorForGivenMessage);
        #print(len(dynamicAttributrMappingDictionary));
        #Commenting for time being to eliminate garbage on cosole output screen
        #print(externalIndex," th vector is ",vectorForGivenMessage," for message "+individualPuncutationRemovedMessage);
    print("message regulsr ",frequencyOfWordsInRegularMessages);
    print("spam regular ",frequencyOfWordsInSpamMessages);

def storeInProbabilityStoreForValue(probabilityHolder,wordToUpdateProbabilityFor):
    if (wordToUpdateProbabilityFor in probabilityHolder):
        probabilityHolder[wordToUpdateProbabilityFor]+=1;
    else:
        probabilityHolder[wordToUpdateProbabilityFor]=1;
            
getAndFilterMessagesInDataStructureWithFileName('SMSSpamCollection');

