from FileReader import *
from StringUtilities import *
from collections import Counter
import operator
import re

def getAndFilterMessagesInDataStructureWithFileName(inputFileName):
    separatedMessageData= convertAndGetFilesDataInListFromFileWithName(inputFileName);
    dynamicAttributrMappingDictionary={};
    dynamicAttributrMappingDictionary['numberOfCapitalLettersInMessage']=1;
    dynamicAttributrMappingDictionary['lengthOfLongestAllCapitalword']=2;
    dynamicAttributrMappingDictionary['messageTypeIndicator']=3;
    dynamicAttributrMappingDictionary['lengthOfLongestWordInString']=4;

    fullMessageDetailsWithRegularAndSpamMessage=separatedMessageData[0]+separatedMessageData[1];

    

    #print(len(fullMessageDetailsWithRegularAndSpamMessage));
    filteredMessagesWithPunctuationElimination=[];

    collectionOfVectorsOfAllStatements=[];
    #Dictionaries which store frequency of each word occurring in each actual message and spam respectively
    #This will later be used for Naive Bayes inference
    frequencyOfWordsInRegularMessages={};
    frequencyOfWordsInSpamMessages={};
    
    for i,individualMessage in enumerate(fullMessageDetailsWithRegularAndSpamMessage):
        filteredMessagesWithPunctuationElimination.append(getStringWithPunctuationsRemoved(individualMessage));


    for externalIndex,individualPuncutationRemovedMessage in enumerate(filteredMessagesWithPunctuationElimination):
        tokenizedDataIntoArray=list(filter(None, re.split('\W+',individualPuncutationRemovedMessage)));
        #when message is full of punctuations like e.g. :-) we get None after removing them which makes resulting string totally unusable

        #if not tokenizedDataIntoArray:
        #    continue;

        vectorForGivenMessage=[-1]*len(dynamicAttributrMappingDictionary);
        
        #initialize these variable for each message that will be encountered
        
        lengthOfLongestAllCapitalword=-1;

        for index,individualAttributeTokens in enumerate(tokenizedDataIntoArray):


            lengthOfCurrentSelectedToken=len(individualAttributeTokens);
            
            #if(lengthOfCurrentSelectedToken<3):
            #    continue;
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
                vectorForGivenMessage[dynamicAttributrMappingDictionary['messageTypeIndicator']-1]='m';
                storeInProbabilityStoreForValue(frequencyOfWordsInRegularMessages,individualAttributeTokens)
            else:
                #We have spam as we already crossed message boundary
                vectorForGivenMessage[dynamicAttributrMappingDictionary['messageTypeIndicator']-1]='s';
                storeInProbabilityStoreForValue(frequencyOfWordsInSpamMessages,individualAttributeTokens)
         #Here end tokenizer loop and we go get next message - Might it be a spam or not   
        vectorForGivenMessage[dynamicAttributrMappingDictionary['numberOfCapitalLettersInMessage']-1]=getNumberOfCapitalLettersFromWord(individualPuncutationRemovedMessage);
        #print(tokenizedDataIntoArray," sequence tokens ");
        if tokenizedDataIntoArray:
            vectorForGivenMessage[dynamicAttributrMappingDictionary['lengthOfLongestWordInString']-1]=len(max(tokenizedDataIntoArray, key=len));
            collectionOfVectorsOfAllStatements.append(vectorForGivenMessage);
        #print(len(dynamicAttributrMappingDictionary));
        #Commenting for time being to eliminate garbage on cosole output screen
        #print(externalIndex," th vector is ",vectorForGivenMessage," for message "+individualPuncutationRemovedMessage);

    #We got out of even the most extrenal loop possible 

    #topNMostOccurringWordsInMessages=dict(Counter(frequencyOfWordsInRegularMessages).most_common(5));
    #topNMostOccurringWordsInSpams=dict(Counter(frequencyOfWordsInSpamMessages).most_common(5));
    #return frequencyOfWordsInRegularMessages,frequencyOfWordsInSpamMessages;
    #print("maessage most occurring n words ",frequencyOfWordsInRegularMessages);
    #print("spam most occurring n words ",frequencyOfWordsInSpamMessages);
    return frequencyOfWordsInRegularMessages,frequencyOfWordsInSpamMessages;
    

#Warning - To use in future to counteract against differences in vector lengths
#element=(currentIndex > length(vectorUnderConsideration)-1)?-1:vectorUnderConsideration[currentIndex]
#will be used in kMean clustering

def storeInProbabilityStoreForValue(probabilityHolder,wordToUpdateProbabilityFor):
    if (wordToUpdateProbabilityFor in probabilityHolder):
        probabilityHolder[wordToUpdateProbabilityFor]+=1;
    else:
        probabilityHolder[wordToUpdateProbabilityFor]=1;
                
frequencyOfAllWordsInGivenMessagesCorpse =[];
frequencyOfAllWordsInGivenMessagesCorpse=getAndFilterMessagesInDataStructureWithFileName('SMSSpamCollection');
print(frequencyOfAllWordsInGivenMessagesCorpse);

