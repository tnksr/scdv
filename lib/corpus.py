# coding: utf-8
import sys

class Corpus(object):

    def __init__(self, min_length=3):
        self.indexs = []
        self.documents = []
        self.min_length = min_length
        self.vocabulary = {}

    def build(self, input_file: str, tokenizer: object, delimiter='\t'):
        # shopid_to_topic
        with open(input_file) as f:
            for line in f:
                line = line.split(delimiter)
                if len(line) < 2: continue
                text_id = line[0]
                text = tokenizer(' '.join(text[1:]))
                if len(text) < self.min_length: continue
                if text in self.documents: continue
                self.indexs.append(text_id)
                self.documents.append(text)
                for word in text:
                    if self.vocabulary.get(word) is None:
                        self.vocabulary[word] = len(self.vocabulary)