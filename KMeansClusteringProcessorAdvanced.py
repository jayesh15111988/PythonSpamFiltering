import random;
import math;
from FileReader import *;
import Constants;
from Constants import *;
import os.path;
from dateTimeUtility import isTrainingFileOldEnough;
from AlgorithmVerificationRoutine import getMessagesAndClassLablesWithInputFile;

def GenerateKMeansClusters(collectionOfVectorsOfAllMessages,numberOfDesiredOutputClusters,collectionOfVectorsOfProductionMessages,inputProductionMessagesFileName):

    #Beginning of computation for K-means clustering for given set of message - Gotta be challenging
    numberOfClusters=2 #Since there are only two classes for use viz. Spam and Regular Messages
    
    centroidHolderForInputMessages=[];
    numberOfClustersGeneratedSoFar=0;
    duplicateChecker={};
    messageCategorizationData=[];
    maximumVectorSequenceValue=len(collectionOfVectorsOfAllMessages)-1
        
    #If we already populated centroid holder data, use it instead of generating all over again
    #Check if file with pre calculated data exists. We are checking for only one file as both files get generated altogether

    if(os.path.isfile(Constants.OUTPUT_CENTROID_FILE_NAME)):
        centroidHolderForInputMessages=getDataStructureFromFileWithName(Constants.OUTPUT_CENTROID_FILE_NAME);
        messageCategorizationData=getDataStructureFromFileWithName(Constants.OUTPUT_MESSAGE_CATEGORIZATION_FILE_NAME); 
    


    if(len(centroidHolderForInputMessages)==0 or isTrainingFileOldEnough(Constants.OUTPUT_CENTROID_FILE_NAME)):
        while(numberOfClustersGeneratedSoFar<numberOfDesiredOutputClusters):

            #generate random point first range is
            #We will take any random point from collection of points as a centroid

            randomVectorSequence=random.randint(0, maximumVectorSequenceValue);
        
            #Err, we regenerated same data point. Try again with another random function run

            if(randomVectorSequence in duplicateChecker):
                continue;
            else:
                centroidHolderForInputMessages.append(collectionOfVectorsOfAllMessages[randomVectorSequence]);
                numberOfClustersGeneratedSoFar=numberOfClustersGeneratedSoFar+1;
                duplicateChecker[randomVectorSequence]=1;
        #Both centroids thus generated, append -1 to them until for both of them length is equal to
        #maximum length of vector in given collection
        lengthOfMaximumLengthVectorInCollection=len(max(collectionOfVectorsOfAllMessages, key=len));  #len(collectionOfVectorsOfAllMessages[maximumVectorSequenceValue]);

        newCentroidHolderForInputMessages=[];
        for individualCentroids in centroidHolderForInputMessages:
            while(len(individualCentroids)<lengthOfMaximumLengthVectorInCollection):
                individualCentroids.append(Constants.DEFAULT_CENTROID_TO_ASSIGN_TO_DATA_POINT);

        #We made sure, it won't cause index out of bound for an incomplete centroid vectors
        #In case of Other vectors, we will simple check against index and their length.
        # If desired index >(=)len(vector)-1(len(vector)) then don't access elemnt simply return -1 as default value
        #Make sure we run this loop until entire sum of difference between previous and current centroid is less
        #than some predefined threshold  - In other words, all of them have been stabilized for sure to specific set of centroid

        predefinedThreshold=0.5;

        #Initializing distance between centroids
        differenceBetweenCentroids=1000;

        index=2;    
    
        while(differenceBetweenCentroids>predefinedThreshold):
            
            regularMessagesVectorContainer=[];
            spamMessagesVectorContainer=[];
            allMessagesContainer=[regularMessagesVectorContainer,spamMessagesVectorContainer];    
            
            for individualVector in collectionOfVectorsOfAllMessages:
                minDistanceFromCentroid=Constants.DEFAULT_MINIMUM_DISTANCE_FROM_CENTROID;
                centroidNumberToChoose=Constants.DEFAULT_CENTROID_TO_ASSIGN_TO_DATA_POINT;
                euclideanDistanceFromCentroidForGivenVector=0;
                for outerIndex,individualCentroidHolderForInputMessages in enumerate(centroidHolderForInputMessages):
                    for index,individualFeatureValueInFeatureVector in enumerate(individualCentroidHolderForInputMessages):
                    
                        individualVectorAttributeValue=individualVector[index] if index < len(individualVector) else -1
                        individualCentroidPointValue=individualCentroidHolderForInputMessages[index];
                                       
                        euclideanDistanceFromCentroidForGivenVector=euclideanDistanceFromCentroidForGivenVector+(individualCentroidPointValue- individualVectorAttributeValue)**2       
                    euclideanDistanceFromCentroidForGivenVector=(euclideanDistanceFromCentroidForGivenVector)**(0.5);
                    if(euclideanDistanceFromCentroidForGivenVector<minDistanceFromCentroid):
                        minDistanceFromCentroid=euclideanDistanceFromCentroidForGivenVector;
                        centroidNumberToChoose=outerIndex;
                if(centroidNumberToChoose!=-1):
                    allMessagesContainer[centroidNumberToChoose].append(individualVector);
            #Calculate average of all points in respective centroid holders and reassign them as centroids
            index=index-1;
    
        #Previous centroid that we had - Now calculate new centroid from the array we have and compare new centroid with older ones

            lengthOfVector=0;
            newCentroidHolderForPoints=[];
            lengthOfVector=lengthOfMaximumLengthVectorInCollection;
            for index,individualCalculatedVector in enumerate(allMessagesContainer):
                tempHolderArrayForCentroids=[];
                lengthOfVectorInVector=len(individualCalculatedVector);
                vectorWithAverageValueInit=[];
                for indexLevelOne in range(0,lengthOfVector):
                    totalValue=0
                    for indexLevelTwo,individualPointVector in enumerate(individualCalculatedVector):
                        valToAdd=individualPointVector[indexLevelOne] if indexLevelOne < len(individualPointVector) else -1;
                        totalValue=totalValue+valToAdd;
                    vectorWithAverageValueInit.append(totalValue/lengthOfVectorInVector);
                newCentroidHolderForPoints.append(vectorWithAverageValueInit);
                


            #Now calculate difference between old and new centroids
            #Now Append -1 to empty centroid value
        
            sumOfDifferencesForCentroids=0.0;
            for indexOuttermost in range(0,numberOfDesiredOutputClusters):
                oldIndividualCentroid=centroidHolderForInputMessages[indexOuttermost];
                newIndividualCentroid=newCentroidHolderForPoints[indexOuttermost];
                for indexInIndividualVector in range(0,lengthOfVector):
                    sumOfDifferencesForCentroids=sumOfDifferencesForCentroids+math.fabs(oldIndividualCentroid[indexInIndividualVector]-newIndividualCentroid[indexInIndividualVector]);        
    
            differenceBetweenCentroids=sumOfDifferencesForCentroids;
            print("Difference between set of points for current and previous values of"
                  "Centroids is  ",differenceBetweenCentroids);
            centroidHolderForInputMessages=newCentroidHolderForPoints;

        #Filter all generated points to determine which section correspond to what
        numberOfSpamsInCluster=0;
        numberOfActualMessagesInCluster=0;

        for individualMessageVectorFromCluster in allMessagesContainer:
            for individualVectorInVectorsCollection in individualMessageVectorFromCluster:
                if(individualVectorInVectorsCollection[2]==0):
                    numberOfSpamsInCluster=numberOfSpamsInCluster+1;
                else:
                    numberOfActualMessagesInCluster=numberOfActualMessagesInCluster+1;
            break;

    
    
        if(numberOfSpamsInCluster>numberOfActualMessagesInCluster):
            writeDataStructureToFileWithName([{0:"spam",1:"regularMessage"}],[Constants.OUTPUT_MESSAGE_CATEGORIZATION_FILE_NAME]);
        else:
            writeDataStructureToFileWithName([{1:"spam",0:"regularMessage"}],[Constants.OUTPUT_MESSAGE_CATEGORIZATION_FILE_NAME]);
        messageCategorizationData=getDataStructureFromFileWithName(Constants.OUTPUT_MESSAGE_CATEGORIZATION_FILE_NAME);

        writeDataStructureToFileWithName([allMessagesContainer,centroidHolderForInputMessages],[Constants.OUTPUT_VECTOR_FILE_NAME,Constants.OUTPUT_CENTROID_FILE_NAME]);

    #In the previous step, we generated centroid based off of a training data and
    #Now it's time to classify messages from our production data

    #This is array of array to hold Spam and Regular messages
    finalClassificationHolder=[[],[]];

    inputMessageListForKMeansEvaluation=getFileData(inputProductionMessagesFileName);

    #This will come into picture when we will verify model on training data itself
    isTestFileInput=False;

    listOfSpamAndRegularMessagesTokens=[];
    listOfIndividualMessages=[];

    #To verify if this is a testing or production data file
    tokenizedLine=inputMessageListForKMeansEvaluation[0].split('\t');

    if(len(tokenizedLine)>1):
        isTestFileInput=True;
        getMessagesAndClassLablesWithInputFile(inputMessageListForKMeansEvaluation,listOfSpamAndRegularMessagesTokens,
                                               listOfIndividualMessages);


    numberOfCorrectClassifications=0;
    totalNumberOfInputMessages=len(collectionOfVectorsOfProductionMessages);

    for vectorCollectionIndex,individualTestVector in enumerate(collectionOfVectorsOfProductionMessages):
        minDistanceFromCentroids= DEFAULT_MINIMUM_DISTANCE_FROM_CENTROID;
        centroidToAssign=DEFAULT_CENTROID_TO_ASSIGN_TO_DATA_POINT;
        for centroidHolderIndex,individualFinalCentroidValues in enumerate(centroidHolderForInputMessages):
            maxVectorLength=len(individualFinalCentroidValues);
            tempminDistanceFromCentroids=0.0;
            for individualIndex in range(0,maxVectorLength):
                individualVectorPointValue=individualTestVector[individualIndex] if individualIndex < len(individualTestVector) else Constants.DEFAULT_CENTROID_TO_ASSIGN_TO_DATA_POINT
                tempminDistanceFromCentroids=tempminDistanceFromCentroids+(individualVectorPointValue-individualFinalCentroidValues[individualIndex])**2;
            tempminDistanceFromCentroids=(tempminDistanceFromCentroids)**0.5;
            if(tempminDistanceFromCentroids<minDistanceFromCentroids):
                minDistanceFromCentroids=tempminDistanceFromCentroids;
                centroidToAssign=centroidHolderIndex;
        finalClassificationHolder[centroidToAssign].append(individualTestVector);
        isCurrentInputMessageSpam=isMessageSpam(messageCategorizationData[centroidToAssign]);



        if(isTestFileInput):
            print("Input Message is ",listOfIndividualMessages[vectorCollectionIndex] ," And Spam indicator is -->  ",isCurrentInputMessageSpam);
            if(isCurrentInputMessageSpam == listOfSpamAndRegularMessagesTokens[vectorCollectionIndex]):
                numberOfCorrectClassifications=numberOfCorrectClassifications+1;

    if(isTestFileInput):
        print("Total Messages Parsed ",totalNumberOfInputMessages,"Number of Correct Classifications",numberOfCorrectClassifications,
          " Accuracy of K Means classification ",numberOfCorrectClassifications/totalNumberOfInputMessages);


def isMessageSpam(messageLabelName):
    messageSpamIndictor=False;
    if(messageLabelName==Constants.DEFAULT_SPAM_INDICATOR_STRING):
        messageSpamIndictor=True;
    return messageSpamIndictor;

#Sample small size data to verify correctness of an implementation
#GenerateKMeansClusters([[1,2,0,3,4,3,6,8,9,0],[6,5,0,4,3,2,1],[4,5,0,6,3,2,1],[6,5,0,4,1,2,3],
# [100,200,1,300,456,432,291],[300,400,1,200,100,500,300]],2,[[1,3,4,5,6,7,5]]);









                
                
            
            
