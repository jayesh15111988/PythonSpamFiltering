from FileReader import *
from StringUtilities import *
from collections import Counter
import operator
import re



def getAndFilterMessagesInDataStructureWithFileName(inputFileName,frequencyOfWordsInRegularMessages,frequencyOfWordsInSpamMessages,collectionOfVectorsOfAllStatements):

    separatedMessageData= convertAndGetFilesDataInListFromFileWithName(inputFileName);

    numberOfActualMessages=len(separatedMessageData[0]);
    numberOfSpamMessages=len(separatedMessageData[1]);
    #print(numberOfActualMessages," Initial ",numberOfSpamMessages);
    dynamicAttributrMappingDictionary={};
    dynamicAttributrMappingDictionary['numberOfCapitalLettersInMessage']=1;
    dynamicAttributrMappingDictionary['lengthOfLongestAllCapitalword']=2;
    dynamicAttributrMappingDictionary['messageTypeIndicator']=3;
    dynamicAttributrMappingDictionary['lengthOfLongestWordInString']=4;

    fullMessageDetailsWithRegularAndSpamMessage=separatedMessageData[0]+separatedMessageData[1];

    

    #print(len(fullMessageDetailsWithRegularAndSpamMessage));
    filteredMessagesWithPunctuationElimination=[];

    
    #Dictionaries which store frequency of each word occurring in each actual message and spam respectively
    #This will later be used for Naive Bayes inference
    #frequencyOfWordsInRegularMessages={};
    #frequencyOfWordsInSpamMessages={};
    
    for i,individualMessage in enumerate(fullMessageDetailsWithRegularAndSpamMessage):
        filteredMessagesWithPunctuationElimination.append(getStringWithPunctuationsRemoved(individualMessage));


    for externalIndex,individualPuncutationRemovedMessage in enumerate(filteredMessagesWithPunctuationElimination):
        tokenizedDataIntoArray=list(filter(None, re.split('\W+',individualPuncutationRemovedMessage)));
        #when message is full of punctuations like e.g. :-) we get None after removing them which makes resulting string totally unusable

        if not tokenizedDataIntoArray:
            if(externalIndex<len(separatedMessageData[0])):
                numberOfActualMessages=numberOfActualMessages-1;
            else:
                numberOfSpamMessages=numberOfSpamMessages-1;
            continue;

        vectorForGivenMessage=[-1]*len(dynamicAttributrMappingDictionary);
        
        #initialize these variable for each message that will be encountered
        #print(externalIndex);
        lengthOfLongestAllCapitalword=-1;

        for index,individualAttributeTokens in enumerate(tokenizedDataIntoArray):


            lengthOfCurrentSelectedToken=len(individualAttributeTokens);
            
            if(lengthOfCurrentSelectedToken<3):
                continue;
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
        #if tokenizedDataIntoArray:
        vectorForGivenMessage[dynamicAttributrMappingDictionary['lengthOfLongestWordInString']-1]=len(max(tokenizedDataIntoArray, key=len));
        collectionOfVectorsOfAllStatements.append(vectorForGivenMessage);
        #print(len(dynamicAttributrMappingDictionary));
        #Commenting for time being to eliminate garbage on cosole output screen
        #print(externalIndex," th vector is ",vectorForGivenMessage," for message "+individualPuncutationRemovedMessage);
    #Last two value for 'collectionOfVectorsOfAllStatements' we append number of valid messages and spams respectively
    #print(numberOfActualMessages," Final ",numberOfSpamMessages)
    collectionOfVectorsOfAllStatements.append(numberOfActualMessages);
    collectionOfVectorsOfAllStatements.append(numberOfSpamMessages);
    #We got out of even the most extrenal loop possible 
    #print(len(frequencyOfWordsInRegularMessages)," fucking python ");
    #topNMostOccurringWordsInMessages=dict(Counter(frequencyOfWordsInRegularMessages).most_common(5));
    #topNMostOccurringWordsInSpams=dict(Counter(frequencyOfWordsInSpamMessages).most_common(5));
    #return frequencyOfWordsInRegularMessages,frequencyOfWordsInSpamMessages;
    #print("maessage most occurring n words ",frequencyOfWordsInRegularMessages);
    #print("spam most occurring n words ",frequencyOfWordsInSpamMessages);
    #return frequencyOfWordsInRegularMessages,frequencyOfWordsInSpamMessages;
    

#Warning - To use in future to counteract against differences in vector lengths
#element=(currentIndex > length(vectorUnderConsideration)-1)?-1:vectorUnderConsideration[currentIndex]
#will be used in kMean clustering

def storeInProbabilityStoreForValue(probabilityHolder,wordToUpdateProbabilityFor):
    if (wordToUpdateProbabilityFor in probabilityHolder):
        probabilityHolder[wordToUpdateProbabilityFor]+=1;
    else:
        probabilityHolder[wordToUpdateProbabilityFor]=1;
                

def setProbabilityForOccurrenceOfEachWordInStore(inputStoreWithCountOfEachWord):
    totalFrequencyOfAllWordsInStore=0;
    collectionOfAllWordsInStore=inputStoreWithCountOfEachWord.keys();

    for individualKey in collectionOfAllWordsInStore:
        totalFrequencyOfAllWordsInStore=totalFrequencyOfAllWordsInStore+inputStoreWithCountOfEachWord[individualKey];

    
    for word in collectionOfAllWordsInStore:
        inputStoreWithCountOfEachWord[word]=inputStoreWithCountOfEachWord[word]/totalFrequencyOfAllWordsInStore;
       
def isMessageSpam(receivedMessage,frequencyOfWordsInRegularMessages,frequencyOfWordsInSpamMessages,probabilityOfRegularMessage,probabilityOfSpamMessage):
    tokenizedInputString = re.split('\W+',getStringWithPunctuationsRemoved(receivedMessage));

    #Initialize probability values first
    temporaryProbabilityThatMessageIsSpam=1.0;
    temporaryProbabilityThatMessageIsNotSpam=1.0;
    #print(frequencyOfWordsInRegularMessages,"  ",frequencyOfWordsInSpamMessages);
    #Now using Actual Bayes Theorm;

    #This is shady - They say use zero or any marginal value using minimum value among them all
    lowestProbabilityValue=(min(min(frequencyOfWordsInRegularMessages.values()),min(frequencyOfWordsInSpamMessages.values())))/10;
    

    print("Lowest ",lowestProbabilityValue);
    
    for individualTokenInInputMessage in tokenizedInputString:
        if not(individualTokenInInputMessage in frequencyOfWordsInRegularMessages):
            temporaryProbabilityThatMessageIsNotSpam=temporaryProbabilityThatMessageIsNotSpam*lowestProbabilityValue;
        else:
            temporaryProbabilityThatMessageIsNotSpam=temporaryProbabilityThatMessageIsNotSpam*frequencyOfWordsInRegularMessages[individualTokenInInputMessage];
            
        if not(individualTokenInInputMessage in frequencyOfWordsInSpamMessages):
            temporaryProbabilityThatMessageIsSpam=temporaryProbabilityThatMessageIsSpam*lowestProbabilityValue;
        else:
            temporaryProbabilityThatMessageIsSpam=temporaryProbabilityThatMessageIsSpam*frequencyOfWordsInSpamMessages[individualTokenInInputMessage];     

    finalProbabilityThatMessageIsNotSpam=temporaryProbabilityThatMessageIsNotSpam*probabilityOfRegularMessage;
    finalProbabilityThatMessageIsSpam=temporaryProbabilityThatMessageIsSpam*probabilityOfSpamMessage;
    return finalProbabilityThatMessageIsSpam > finalProbabilityThatMessageIsNotSpam;

frequencyOfWordsInRegularMessages={};
frequencyOfWordsInSpamMessages={};
collectionOfVectorsOfAllStatements=[];

getAndFilterMessagesInDataStructureWithFileName('SMSSpamCollection',frequencyOfWordsInRegularMessages,frequencyOfWordsInSpamMessages,collectionOfVectorsOfAllStatements);

#Last two value of collectionOfVectorsOfAllStatements contains number of actual messages and spams respectively
#P.S. Last value is number of spam messages out of total received message corpse


setProbabilityForOccurrenceOfEachWordInStore(frequencyOfWordsInRegularMessages);
setProbabilityForOccurrenceOfEachWordInStore(frequencyOfWordsInSpamMessages);

#Second last value
totalNumberRegularMessagesAfterFiltering=collectionOfVectorsOfAllStatements[-2];
#Last value
totalNumberSpamMessagesAfterFiltering=collectionOfVectorsOfAllStatements[-1];



probabilityOfRegularMessage=(totalNumberRegularMessagesAfterFiltering/(totalNumberRegularMessagesAfterFiltering+totalNumberSpamMessagesAfterFiltering));
probabilityOfSpamMessage=(totalNumberSpamMessagesAfterFiltering/(totalNumberRegularMessagesAfterFiltering+totalNumberSpamMessagesAfterFiltering));
#print(probabilityOfRegularMessage," **************************  ",probabilityOfSpamMessage);

print(isMessageSpam("Your payment has been scheduled",frequencyOfWordsInRegularMessages,frequencyOfWordsInSpamMessages,probabilityOfRegularMessage,probabilityOfSpamMessage));

