import re;
from FileReader import *;
from datetime import datetime;
from dateTimeUtility import isTrainingFileOldEnough;
from StringUtilities import *;
from collections import Counter;
import Constants;
from KMeansClusteringProcessorAdvanced import GenerateKMeansClusters;
from NaiveBayesProcessor import *;
import operator;


def replaceStringWithIntegerForString(inputStringValue):
    convertedIntegerValueToReturn=1;
    if inputStringValue==Constants.DEFAULT_SPAM_CHARACTER:
        convertedIntegerValueToReturn=0;
    return convertedIntegerValueToReturn;


def filterPointsForPossibleStringData(messageVectorCollection):
    for individualVectorInCollection in messageVectorCollection:
        for index,individualVectorPoint in enumerate(individualVectorInCollection):
            if type(individualVectorPoint) is str:
               individualVectorInCollection[index] = replaceStringWithIntegerForString(individualVectorPoint);

def getAndFilterMessagesInDataStructureWithFileName(inputFileName,frequencyOfWordsInRegularMessages,frequencyOfWordsInSpamMessages,collectionOfVectorsOfAllStatements,dynamicAttributrMappingDictionary):

    separatedMessageData= convertAndGetFilesDataInListFromFileWithName(inputFileName);

    numberOfActualMessages=len(separatedMessageData[0]);
    numberOfSpamMessages=len(separatedMessageData[1]);
    #print(numberOfActualMessages," Initial ",numberOfSpamMessages);
    
    dynamicAttributrMappingDictionary['numberOfCapitalLettersInMessage']=1;
    dynamicAttributrMappingDictionary['lengthOfLongestAllCapitalword']=2;
    dynamicAttributrMappingDictionary['messageTypeIndicator']=3;
    dynamicAttributrMappingDictionary['lengthOfLongestWordInString']=4;

    fullMessageDetailsWithRegularAndSpamMessage=separatedMessageData[0]+separatedMessageData[1];
    filteredMessagesWithPunctuationElimination=[];
    
    #Dictionaries which store frequency of each word occurring in each actual message and spam respectively
    #This will later be used for Naive Bayes inference
    
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

            #This is because we have return realMessage,spamMessage;
            if(externalIndex<len(separatedMessageData[0])):

                #We have actual message

                vectorForGivenMessage[dynamicAttributrMappingDictionary['messageTypeIndicator']-1]=Constants.DEFAULT_MESSAGE_CHARACTER;
                storeInProbabilityStoreForValue(frequencyOfWordsInRegularMessages,individualAttributeTokens)
            else:

                #We have spam as we already crossed message boundary

                vectorForGivenMessage[dynamicAttributrMappingDictionary['messageTypeIndicator']-1]=Constants.DEFAULT_SPAM_CHARACTER;
                storeInProbabilityStoreForValue(frequencyOfWordsInSpamMessages,individualAttributeTokens)
        #Here end tokenizer loop and we go get next message - Might it be a spam or not   
        vectorForGivenMessage[dynamicAttributrMappingDictionary['numberOfCapitalLettersInMessage']-1]=getNumberOfCapitalLettersFromWord(individualPuncutationRemovedMessage);        
        vectorForGivenMessage[dynamicAttributrMappingDictionary['lengthOfLongestWordInString']-1]=len(max(tokenizedDataIntoArray, key=len));
        collectionOfVectorsOfAllStatements.append(vectorForGivenMessage);
        
        #Commenting for time being to eliminate garbage on cosole output screen
        
    #Last two value for 'collectionOfVectorsOfAllStatements' we append number of valid messages and spams respectively
    collectionOfVectorsOfAllStatements.append(numberOfActualMessages);
    collectionOfVectorsOfAllStatements.append(numberOfSpamMessages);

    #We got out of even the most extrenal loop possible 
    #topNMostOccurringWordsInMessages=dict(Counter(frequencyOfWordsInRegularMessages).most_common(5));
    #topNMostOccurringWordsInSpams=dict(Counter(frequencyOfWordsInSpamMessages).most_common(5));

def storeInProbabilityStoreForValue(probabilityHolder,wordToUpdateProbabilityFor):
    if (wordToUpdateProbabilityFor in probabilityHolder):
        probabilityHolder[wordToUpdateProbabilityFor]+=1;
    else:
        probabilityHolder[wordToUpdateProbabilityFor]=1;
                
frequencyOfWordsInRegularMessages={};
frequencyOfWordsInSpamMessages={};
collectionOfVectorsOfAllMessages=[];
dynamicAttributrMappingDictionary={};

print("filr name is ",Constants.SMALL_TRAINING_DATA_FILE);
getAndFilterMessagesInDataStructureWithFileName(Constants.SMALL_TRAINING_DATA_FILE,frequencyOfWordsInRegularMessages,frequencyOfWordsInSpamMessages,collectionOfVectorsOfAllMessages,dynamicAttributrMappingDictionary);


def runNaiveBayesOnDataFromFileWithName(sampleSpamFilename,collectionOfVectorsOfAllMessages):

    #Commenting for time being as Naive Bayes is on hold temporarily

    #Last two value of collectionOfVectorsOfAllStatements contains number of actual messages and spams respectively
    #P.S. Last value is number of spam messages out of total received message corpse

    ##### Computation for classifying any future message as regular/spam using Bayes Theorm #####
    #For each word, we set proobability of occurrence in each spam and non-spam category
    global frequencyOfWordsInRegularMessages;
    global frequencyOfWordsInSpamMessages;
    
    print(frequencyOfWordsInRegularMessages, " Frequency in regular message ");
    if(isTrainingFileOldEnough(Constants.OUTPUT_REGULAR_MESSAGES_WORD_FREQUENCY) or isTrainingFileOldEnough(Constants.OUTPUT_SPAM_MESSAGES_WORD_FREQUENCY)):
        print("Storing first time");
        setProbabilityForOccurrenceOfEachWordInStore(frequencyOfWordsInRegularMessages);
        setProbabilityForOccurrenceOfEachWordInStore(frequencyOfWordsInSpamMessages);
        writeDataStructureToFileWithName([frequencyOfWordsInRegularMessages],[Constants.OUTPUT_REGULAR_MESSAGES_WORD_FREQUENCY]);
        writeDataStructureToFileWithName([frequencyOfWordsInSpamMessages],[Constants.OUTPUT_SPAM_MESSAGES_WORD_FREQUENCY]);
    else:
        print("utilizing stored data");
        frequencyOfWordsInRegularMessages=getDataStructureFromFileWithName(Constants.OUTPUT_REGULAR_MESSAGES_WORD_FREQUENCY);
        frequencyOfWordsInSpamMessages=getDataStructureFromFileWithName(Constants.OUTPUT_SPAM_MESSAGES_WORD_FREQUENCY);
    #Second last value - Total number of regular messages in given corpse
    print("Frequency in regular Message ",frequencyOfWordsInRegularMessages,"Frequency in spam message ",frequencyOfWordsInSpamMessages);

    
        
    
        



    
    totalNumberRegularMessagesAfterFiltering=collectionOfVectorsOfAllMessages[-2];

    #Last value - Total number of spams
    totalNumberSpamMessagesAfterFiltering=collectionOfVectorsOfAllMessages[-1];

    probabilityOfRegularMessage=(totalNumberRegularMessagesAfterFiltering/(totalNumberRegularMessagesAfterFiltering+totalNumberSpamMessagesAfterFiltering));
    probabilityOfSpamMessage=(totalNumberSpamMessagesAfterFiltering/(totalNumberRegularMessagesAfterFiltering+totalNumberSpamMessagesAfterFiltering));

    inputMessageListForNaiveBayesEvaluation=getFileData(sampleSpamFilename);
    for individualProductionMessage in inputMessageListForNaiveBayesEvaluation:
        print("Message ->>  ",individualProductionMessage," Is Message Spam or not indicator -->> ",isMessageSpam(individualProductionMessage,frequencyOfWordsInRegularMessages,frequencyOfWordsInSpamMessages,probabilityOfRegularMessage,probabilityOfSpamMessage));

def runKMeansClusteringOnDataFromFileWithName(sampleSpamFilename,collectionOfVectorsOfAllMessages):
    
    del collectionOfVectorsOfAllMessages[-2:];
    filterPointsForPossibleStringData(collectionOfVectorsOfAllMessages);
    collectionOfVectorsOfProductionMessages=[];
    getAndFilterMessagesInDataStructureWithFileName(sampleSpamFilename,frequencyOfWordsInRegularMessages,frequencyOfWordsInSpamMessages,collectionOfVectorsOfProductionMessages,dynamicAttributrMappingDictionary);

    #Remove last two elements which carry no meaning with them

    del collectionOfVectorsOfProductionMessages[-2:];
    filterPointsForPossibleStringData(collectionOfVectorsOfProductionMessages);
    
    #This is collection of vectors converted from user data - Pass it to K-means to easily verify thereafter
    #Length is 6 and not just 4 as per number of lines, as we also append number of (approximate) regular and spam messages
    
    GenerateKMeansClusters(collectionOfVectorsOfAllMessages,2,collectionOfVectorsOfProductionMessages);
    print("Collection of messages is ",len(max(collectionOfVectorsOfAllMessages, key=len)));
    print("length of collection vector after filtering string characters is ",len(collectionOfVectorsOfAllMessages));
#print("collection is ",collectionOfVectorsOfAllMessages);

startTime = datetime.now();
runNaiveBayesOnDataFromFileWithName(Constants.PRODUCTION_DATA_FILE,collectionOfVectorsOfAllMessages);
#print("Message freq",getDataStructureFromFileWithName(Constants.OUTPUT_REGULAR_MESSAGES_WORD_FREQUENCY),"\n\n");
#print("Spams Freq",getDataStructureFromFileWithName(Constants.OUTPUT_SPAM_MESSAGES_WORD_FREQUENCY),"\n\n");
print(datetime.now()-startTime);
#runKMeansClusteringOnDataFromFileWithName(Constants.PRODUCTION_DATA_FILE,collectionOfVectorsOfAllMessages);














