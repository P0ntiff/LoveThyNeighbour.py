#k-nearest neighbours model for language classification

import csv
import re
from string import digits
import numpy as np
import math
import time

#Encoding=utf8  
import sys  
reload(sys)  
sys.setdefaultencoding('utf8')
startTime = time.time()

languageLabels = {'0': "Slovak",
				  '1': "French", 
				  '2': "Spanish", 
				  '3': "German", 
				  '4': "Polish"}

#How many examples in each language
exampleCounts = {'0': 14167,
		  		 '1': 141283,
		  		 '2': 69974,
		  		 '3': 37014,
		  		 '4': 14079}

#Language predicted for empty lines (French as it is the most common)
defaultPrediction = 1

#The number of nearest neighbours to pick the majority class from (i.e the "k" in kNN)
k = 10

#Percentage of training set for validation (i.e 100(1 - valPercent)% of the training data is used for training)
valPercent = 0.99

#Extract second column of train_set_x.csv as "examples", etc for labels and testData
examples = [item[1] for item in list(csv.reader(open("train_set_x.csv", 'rb')))[1:]]
labels = [item[1] for item in list(csv.reader(open("train_set_y.csv", 'rb')))[1:]]
testData = [item[1] for item in list(csv.reader(open("test_set_x.csv", 'rb')))[1:]]

#Put aside some data for validation/preventing overfitting
dividor = int(len(examples) * valPercent)

#TODO: Implement code on validation examples (and fix vectoriseSentence taking 20 highest chars)
valExamples = [examples[i] for i in range(0, dividor)]
valLabels = [labels[i] for i in range(0, dividor)]

trainExamples = [examples[i] for i in range(dividor, len(examples))]
trainLabels = [labels[i] for i in range(dividor, len(examples))]

#Outputs a list of classifications/predictions for each example in the test set
def main(rawExamples, trainingLabels, rawTestData):
	#Preprocessing the training data
	examples = examplePreprocessing(rawExamples, False)
	#Training: -Building tf-idf scores for each character in each example
	#          -Converting each example to a 20-long list of the 20 highest tf-idf scores for the sentence
	trainingVectors, documentFrequencies = buildTF_IDFScores(examples, trainingLabels)

	#Preprocessing the testing data
	testData = examplePreprocessing(rawTestData, True)
	#Testing:  -Getting 5 vectors for each example (one for each possible language of the testing example (as the actual label is unknown))
	#          -Calculating the k-nearest neighbours for each possibility
	predictions = []
	print("...Calculating nearest neighbours")
	for i in range(0, len(testData)):
		prediction = ""
		testVectors = vectoriseTestSentence(testData[i], documentFrequencies)
		#Empty line indicated by [-1] return
		if testVectors == [-1]:
			prediction = str(defaultPrediction)
		else:
			prediction = kNearestNeighbours(testVectors, trainingVectors)
		predictions.append(prediction)
	print("Calculations complete.")

	#Output
	with open('knnPredictions.csv', 'wb') as output:
		writer = csv.writer(output)
		writer.writerow(['ID', 'Category'])
		for i in range(0, len(predictions)):
			writer.writerow([i, predictions[i]])

	#DEBUG
	#for (key, value) in tf_idf_scores.items():
	#	for j in value:
	#		print(key + " - " + str(j))
	#TODO: Come up with distance metric

	print (" --- Runtime (seconds): ~" + str((time.time() - startTime)) + " --- ")
	return None

#Distance metric: Cosine similarity between test example vectors and training example vectors
#Returns a list of (length k) of the cosine similarities with the k-nearest neighbours
#Test Vectors is a list of the vectors generated for each language for this test example
def kNearestNeighbours(testVectors, trainingVectors):
	prediction = "-1"
	globalSimilarities = []
	for l in range(0, 5):
		label = str(l)
		localSimilarities = []
		for j in range(0, len(trainingVectors[label])):
			similarity = cosineSimilarity(testVectors[l], trainingVectors[label][j])
		prediction = str(l)

		localSimilarities.append(similarity)

	return prediction


#The distance metric for comparing a test example with training examples
def cosineSimilarity(testVector, trainingVector):
	#a = np.vector(testVector)
	#b = np.vector(trainingVector)


	return None

#Remove digits, extra spaces, emojis, and URLS in the example
#Flatten the string into a list of characters
#Note: Empty line examples are not removed at this step.
#TODO: remove "-", "_", "."  --> Thought: does removing apostrophes lose information? i.e French
#"testing" --> = 0 if training, 1 if testing
def examplePreprocessing(examples, testing):
	cleanExamples = []
#	cleanLabels = []
	#A filter to get rid of all non-multilingual unicode characters (i.e emoji and others)
	characterFilter = re.compile(u"[^\U00000000-\U0000d7ff\U0000e000-\U0000ffff]", flags=re.UNICODE)

	print("...Preprocessing...")
	for i in range(0, len(examples)):
		noEmoji = characterFilter.sub('', unicode(examples[i]))
		flattened = ''.join(noEmoji.split())
		noDigits = str(flattened).translate(None, digits)
		#Check for URLS (check for "http" and if not found check for "www")
		noURLS = ""
		if testing:
			noURLS = noDigits
		else:
			indexOfURL = noDigits.find("http")
			if indexOfURL == -1:
				indexOfURL = noDigits.find("www")
			if indexOfURL != -1:
				noURLS = noDigits[:indexOfURL]
			else:
				noURLS = noDigits

		cleanExamples.append(noURLS)
		#print(str(cleanLabels[i]) + "- " + str(cleanExamples[i]))

	print("Preprocessing complete.")
	return cleanExamples


#Goes through and builds the tf-idf scores for every character in each language
#Outputs a list of five dictionaries (one for each language) where each list
# contains lists of the 20-highest tf-idf scores for the sentence 
#Precondition: examples is a list of sentences sanitised by examplePreprocessing()
def buildTF_IDFScores(examples, labels):
	print("...Converting training examples to tf-idf vectors....")
	#A count of the total number of examples that contain the character
	documentFrequencies = {"0": {}, "1": {}, "2": {}, "3": {}, "4": {} }
	sentenceCount = len(examples)
	for i in range(0, sentenceCount):
		sentenceLength = len(examples[i])
		label = labels[i]
		foundChars = []
		for j in range(0, sentenceLength):
			char = examples[i][j]
			#Only want to store the first occurence of a character in a sentence
			if (char in foundChars):
				continue
			else:
				foundChars.append(char)
			if (char in documentFrequencies[label].keys()):
				documentFrequencies[label][char] += 1
			else:
				documentFrequencies[label][char] = 1

	listsOfTF_IDFS = {"0": [], "1": [], "2": [], "3": [], "4": []}
	#Get the char's frequency in each sentence and calculate TF-IDF for each char

	for i in range(0, sentenceCount):
		#Get a vector containing the sentence's 20 highest TF-IDF scores
		exampleVector = vectoriseSentence(examples[i], labels[i], documentFrequencies, False)
		if exampleVector == [-1]:
			continue
		#sort in descending order and return
		listsOfTF_IDFS[labels[i]].append(exampleVector)

	print("Training conversion complete.")
	return (listsOfTF_IDFS, documentFrequencies)


#Get the 20 highest character tf-idf scores (for that language) out of the sentence 
#If <20 characters in the training example then 
# 1) get the tf-idf scores of the n characters there
# 2) fill in the rest with the (20 - n) highest tf-idf scored characters for that language
#Precondition: sentence is a flattened list of chars outputted by examplePreprocessing()
#"testing" (boolean) = 0 if training example, 1 if test data
def vectoriseSentence(sentence, label, documentFrequencies, testing):
	sentenceLength = len(sentence)
	if (sentenceLength == 0):
		#store something here?
		#should empty lines be used in training at all?
		return [-1]

	#A count of the total frequency of the character within the sentence
	termFrequencies = {}
	for j in range(0, sentenceLength):
		if (sentence[j] in termFrequencies.keys()):
			termFrequencies[sentence[j]] += 1
		else:
			termFrequencies[sentence[j]] = 1

	#Calculate the list of tf-idfs for the sentence
	tf_idfs = []
	for j in range(0, sentenceLength):
		#TODO: Check if better performance with overall document count instead of per-language example count
		tf_idf = 0
		if testing:
			#Laplace smoothing for division by zero (i.e if this tested character does not appear at all in the language distribution)
			if (sentence[j] not in documentFrequencies[label].keys()):
				tf_idf = termFrequencies[sentence[j]] * math.log(exampleCounts[label] / 1)
			else:
				tf_idf = termFrequencies[sentence[j]] * math.log(exampleCounts[label] / documentFrequencies[label][sentence[j]])
		else:
			tf_idf = termFrequencies[sentence[j]] * math.log(exampleCounts[label] / documentFrequencies[label][sentence[j]])
			#regularise by multiplying by 20/sentenceLength
			tf_idf = tf_idf * (20 / sentenceLength)
		tf_idfs.append(tf_idf)

	#Fill in the rest of the characters with zeros
	#TODO: Think of possible better approach than appending zeros.
	if (sentenceLength < 20):
		for j in range(0, 20 - sentenceLength):
			tf_idfs.append(0)

	#If a training example, take the 20 largest tf-idfs for the vector and output in descending order
	#Otherwise we already have a 20-long vector due to the test set format
	output_tf_idfs = [0 for i in range(0, 20)]
	if testing == False:
		currentMinIndex = 0
		for j in range(0, len(tf_idfs)):
			if tf_idfs[j] > output_tf_idfs[currentMinIndex]:
				output_tf_idfs[currentMinIndex] = tf_idfs[j]
				#update min index
				currMin, currMinIndex = min((val, index) for (index, val) in enumerate(output_tf_idfs))
				currentMinIndex = currMinIndex
		return sorted(output_tf_idfs, reverse=True)
	else:
		return sorted(tf_idfs, reverse=True)

#Return one vector for each language
def vectoriseTestSentence(sentence, documentFrequencies):
	#'vectors' will contain the 5 possible tf-idf vectors (one drawn from each language)
	vectors = []
	for i in range(0, 5):
		vector = vectoriseSentence(sentence, str(i), documentFrequencies, True)
		#if empty line
		if vector == [-1]:
			#TODO: Tell calling function to default to French
			return [-1]
		vectors.append(vector)
	return vectors

def getTF_IDFScore(example, label, index, charIndex):
	return None	

#Repeat by cross-validating and take an average of the predictions
def crossValidationWeighting():
	return None




main(trainExamples, trainLabels, testData)
