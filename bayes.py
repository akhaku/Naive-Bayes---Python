#!/usr/sup/bin/python2.7

""" A Python implementation of the naive-Bayes algorithm. Intended to be run on
    the Tufts Linux servers. To run, run the following command:
    ./soln.py
    The results will be printed to output.csv in the following order:
    n,learn_time,test_time,accuracy
"""

from math import log10
from time import clock

TRAIN_INDEXNAME = "index.Full"
TEST_INDEXNAME = "index.Full"
TRAIN_PATH = "/comp/160/files/project/train/"
TEST_PATH = "/comp/160/files/project/test/"
YES = 0 # Internal representation
NO  = 1 # Internal representation
EMPTY = [-1,0] # Signal that cell in open-addressed hashtable is empty

class HashTable:
    """ A simple hashtable class. Uses open addressing (linear probing) for
        collisions. Does not dynamically resize the hashtable.
    """
    def __init__(self, initSize):
        self.table_size     = initSize
        self.keysSize = initSize
        self.numElems = 0
        self.keys     = [None] * initSize
        self.array    = [None] * initSize
        self.last_lookup = ["",0]

    def get(self, k):
        """ Gets the value at key k """
        index = self._hash(k)
        return self.array[index][1]

    def replace(self, k, v):
        """ Replaces the value of a key already in the table """
        index = self._hash(k)
        self.array[index] = [k, v]

    def delete(self, k):
        """ Deletes the key/value pair at k """
        index = self._hash(k)
        self.array[index] = EMPTY

    def insert(self, k, v):
        """ Inserts (k,v) into the table assuming it's not already there """
        index = self._hash(k, True)
        self.array[index] = [k, v]
        self.keys[self.numElems] = [k, index]
        self.numElems += 1

    def get_val_at_index(self, index):
        """ Shortcut to get table value at index """
        return self.array[index][1]

    def all_keys(self):
        """ Returns the list of all keys that have been inserted into the table,
            including ones that have since been removed. It also returns the
            index associated with that key for a fast lookup.
        """
        return self.keys[0:self.numElems]

    def _hash(self, k, inserting=False):
        """ Double hashing - Jenkins hash function and djb2 """
        if self.last_lookup[0] == k:
            return self.last_lookup[1]
        jHash = 0
        dHash = 5381
        for c in k:
            dHash = ((dHash << 5) + dHash) + ord(c)
            jHash += ord(c)
            jHash += (jHash << 10)
            jHash ^= (jHash >> 6)
        jHash += (jHash << 3)
        jHash ^= (jHash >> 11)
        jHash += (jHash << 15)
        index = (jHash + dHash) % self.table_size
        if inserting:
            while self.array[index] != None and self.array[index][0] != EMPTY:
                index = (index + dHash) % self.table_size
        else:
            if self.array[index] == None: raise KeyError
            while self.array[index][0] != k:
                index = (index + dHash) % self.table_size
                if self.array[index] == None: raise KeyError
        self.last_lookup = [k, index]
        return index

class Classifier:
    """ The base Classifier class. Initialize with the index filename, train
        files path, test files path and an optional n.
        In classCounts, class 0 is yes and 1 is no
        public methods:
            learn: uses the index file from initialization to learn
            classify_doc: Given a filename, classifies it based on learned data
            run_tests: Runs test from index_file provided
    """
    def __init__(self, indexname, train_path, test_path, n=None):
        self.indexname     = indexname
        self.train_path    = train_path
        self.test_path     = test_path
        self.classCounts   = [1, 1] # 1 + num of each class
        self.classFreqs    = [0.0, 0.0]
        self.classWordSize = [0, 0]
        self.vocabSize     = 0
        self.n             = n
        self.table         = HashTable(900001)

    def learn(self):
        """ Use the provided training data to learn """
        index = open(self.indexname, 'r')
        for line in index:
            l         = line.split('|')
            document  = "%s%s.clean" % (self.train_path, l[0])
            clas      = l[1]
            clas      = self._getClass(clas)
            self._add_words(document, clas)
            self.classCounts[clas] += 1
        index.close()
        if self.n >= self.vocabSize: self.n = None
        if self.n is not None: self._get_deltas()
        self._get_class_frequencies()
    
    def classify_doc(self, filename):
        """ Given a string that is a filename, classifies doc based on
            everything learned in training.
        """
        document = open(filename, 'r').read()
        words = document.split()
        yes_score = log10(self.classFreqs[YES])
        no_score  = log10(self.classFreqs[NO])
        for word in words:
            word = word.lower()
            yes_score += log10(self._get_word_freq(word, YES))
            no_score  += log10(self._get_word_freq(word, NO))
        if yes_score > no_score:
            return YES
        else:
            return NO

    def run_tests(self, index_file):
        """ Runs the test suite on the documents given in the index file.
            Returns the percent correct.
        """
        if self.n is not None:
            self.vocabSize = self.n
            newTable = HashTable(900001)
            for d in self.deltas:
                word  = d[0]
                delta = d[1]
                val   = d[2]
                if delta > self.cutoff:
                    newTable.insert(word, val)
            self.table = newTable
        index = open(index_file)
        correct   = 0
        total = 0
        for line in index:
            l        = line.split('|')
            document = "%s%s.clean" % (self.test_path, l[0])
            clas     = l[1]
            clas     = self._getClass(clas)
            detected = self.classify_doc(document)
            if clas == detected:
                correct += 1
            total += 1
        index.close()
        return correct*100.0/total
    
    def _get_class_frequencies(self):
        """ Stores the class frequencies as an instance variable in the class """
        totalPlus2  = self.classCounts[YES] + self.classCounts[NO]
        self.classFreqs[YES] = 1.0 * self.classCounts[YES] / totalPlus2
        self.classFreqs[NO] = 1.0 * self.classCounts[NO] / totalPlus2

    def _get_deltas(self):
        """ Gets deltas for all the words and stores then as a class instance
            variable. Finds the cutoff delta based on the class parameters.
        """
        deltas = [None]*self.vocabSize
        counter = 0
        for word, index in self.table.all_keys():
            val = self.table.get_val_at_index(index)
            delta = abs( log10(self._get_word_freq_fast(val, YES)) - \
                    log10(self._get_word_freq_fast(val, NO)) )
            deltas[counter]=[word, delta, val]
            counter += 1
        self.cutoff = self._nth(deltas, self.vocabSize - self.n + 1)
        self.deltas = deltas

    def _add_words(self, doc, clas):
        """ Adds all the words in doc to our knowledgebase as members of class
            clas.
        """
        f = open(doc, 'r')
        for word in f.read().split():
            word = word.lower()
            self.classWordSize[clas] += 1
            try:
                nums = self.table.get(word)
                nums[clas] += 1
                self.table.replace(word, nums)
            except KeyError:
                self.vocabSize += 1
                nums = [1, 1]
                nums[clas] += 1
                self.table.insert(word, nums)
        f.close()

    def _get_word_freq(self, w, c):
        """ Gets the frequency of word w in class c. If the word doesn't exist,
            return 1
        """
        try:
            freq = 1.0 * self.table.get(w)[c] / (self.classWordSize[c] + self.vocabSize)
        except KeyError:
            freq = 1
        return freq

    def _get_word_freq_fast(self, ar, which):
        """ Gets the frequency given the array of YES occurrences and NO
            occurrences
        """
        freq = 1.0 * ar[which] / (self.classWordSize[which] + self.vocabSize)
        return freq

    def _getClass(self, c):
        """ Implementation of an enum for the two classes """
        if c == "yes":
            c = YES
        else:
            c = NO
        return c

    def _nth(self, sample, n):
        """ Implementation of linear time selection of nth elem """
        pivot = sample[0][1]
        below = [s for s in sample if s[1] < pivot]
        above = [s for s in sample if s[1] > pivot]
        i, j = len(below), len(sample)-len(above)
        if n < i:      return self._nth(below, n)
        elif n >= j:   return self._nth(above, n-j)
        else:          return pivot

def run_classifier(n):
    """ Run the classifier with the parameter n """
    train_index_file = TRAIN_PATH + TRAIN_INDEXNAME
    test_index_file = TEST_PATH + TEST_INDEXNAME
    c = Classifier(train_index_file, TRAIN_PATH, TEST_PATH, n)
    start = clock()
    c.learn()
    after_learned = clock()
    accuracy      =  c.run_tests(test_index_file)
    after_tested  = clock()
    learn_time    = after_learned - start
    testing_time  = after_tested - after_learned
    return "%s,%s,%s,%s\n" % (n, learn_time, testing_time, accuracy)

def main():
    num_datapoints = 100
    init_datapoint = 200
    last_datapoint = 128000
    step_size      = (last_datapoint - init_datapoint)/num_datapoints

    n = init_datapoint
    count = 0
    while n < last_datapoint:
        theResult = run_classifier(n)
        f = open('output.csv','a')
        f.write(theResult)
        f.close()
        n += step_size
        count += 1
main()

