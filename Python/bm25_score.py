import numpy as np
import pandas as pd

class BM25:
    def __init__(self, full_q, d_list, unique=False, k=2, b=0.75):
        """
        Reference1: https://www.elastic.co/blog/practical-bm25-part-2-the-bm25-algorithm-and-its-variables
        Reference2: https://www.youtube.com/watch?v=a3sg6MH8m4k&ab_channel=BadriAdhikari
        """
        self.full_q = full_q
        self.d_list = [d.lower() for d in d_list]
        self.unique = unique
        self.k = k
        self.b = b
        self.avg_doc_len = self.get_avg_len(d_list, unique=unique)
        print(f'Average Document length in corpus: {self.avg_doc_len}')
        self.bm25_scores = []

    @staticmethod
    def get_tf(q, d):
        d_terms = d.split()
        return len([1 for t in d_terms if t == q])

    @staticmethod
    def get_avg_len(d_list, unique=False):
        if unique:
            return np.mean([len(set(d.split())) for d in doc_list])
        return np.mean([len(d.split()) for d in doc_list])

    @staticmethod
    def get_idf(q, d_list):
        df = len([1 for d in d_list if q in d.split()])
        numerator = len(d_list) - df + 0.5
        denominator = df + 0.5
        return np.log(1+(numerator / denominator))

    @staticmethod
    def get_bm25_q(q, d_list, L, unique=False, k=2, b=.75):
        idf_v = BM25.get_idf(q, d_list)
        print(f'IDF for Q: {q} is: {idf_v}')
        bm25_scores = []
        for d in d_list:
            if unique:
                avg_d_l = len(set(d.split())) / L
            else:
                avg_d_l = len(d.split()) / L
            # print(f'Avg_d_l for document: "{d}" is {avg_d_l}')
            tf_v = BM25.get_tf(q, d)
            # print(f'\tTF for: {q} in "{d}" is: {tf_v}')
            numerator = tf_v * (k + 1)
            denominator = tf_v + (k * (1 - b + (b * avg_d_l)))
            bm25_scores.append((idf_v * (numerator / denominator)))
        return bm25_scores

    def get_bm25_scores(self):
        for q in self.full_q.split():
            self.bm25_scores.append(
                self.get_bm25_q(q,
                                self.d_list,
                                L=self.avg_doc_len,
                                unique=self.unique,
                                k=self.k,
                                b=self.b))
        self.bm25_scores = pd.DataFrame(self.bm25_scores).sum().values
        return pd.DataFrame(zip(self.d_list, self.bm25_scores),
                            columns=['Document', 'BM25 Score'])
      
doc_list = [
    "This is a cat",
    "This is a dog",
    "Cat and dog are best friends",
    "This is not a cat but it is a dog",
    "Comeon this is a bull",
    "Cat is an animal Cat are not best friends"
]

bm25_v1_obj = BM25(full_q = 'cat dog', d_list=doc_list, unique=False, k=2, b=0.75)
_tmp = bm25_v1_obj.get_bm25_scores()
_tmp.sort_values('BM25 Score', ascending=False)


"""
We see a variable b which shows up in the denominator and that it’s multiplied by the ratio of the field length we just discussed. 
If b is bigger, the effects of the length of the document compared to the average length are more amplified. 
To see this, you can imagine if you set b to 0, the effect of the length ratio would be completely nullified and 
the length of the document would have no bearing on the score. 
By default, b has a value of 0.75 in Elasticsearch.
"""


"""
A higher/lower k1 value means that the slope of “tf() of BM25” curve changes. 
This has the effect of changing how “terms occurring extra times add extra score.” 
An interpretation of k1 is that for documents of the average length, it is the value of the term frequency that gives a score of half the maximum score for the considered term. 
The curve of the impact of tf on the score grows quickly when tf() ≤ k1 and slower and slower when tf() > k1.

Continuing with our example, with k1 we’re controlling the answer to the question 
“how much more should adding a second ‘shane’ to the document contribute to the score than the first or the third compared to the second?” 
A higher k1 means that the score for each term can continue to go up by relatively more for more instances of that term. 
A value of 0 for k1 would mean that everything except IDF(qi) would cancel out. 
By default, k1 has a value of 1.2 in Elasticsearch.
"""
