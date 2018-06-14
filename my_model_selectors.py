import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN

    L = likelihood of the fitted model
    p = number of free parameters
    N = number of data points
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        num_features = len(self.X[0])
        num_data_points = len(self.X)
        min_bic_score = float('inf')
        best_model = None

        for num_states in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = self.base_model(num_states)
                logL = model.score(self.X, self.lengths)
                num_free_params = num_states ** 2 + 2 * num_states * num_features - 1
                bic_score = -2 * logL + num_free_params * np.log(num_data_points)
                if bic_score < min_bic_score:
                    min_bic_score = bic_score
                    best_model = model

            except Exception:
                return best_model

        return best_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        total_quantity_words = len(self.hwords)
        max_dic_score = float('-inf')
        best_model = None

        for num_states in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = self.base_model(num_states)
                current_word_logL = model.score(self.X, self.lengths)
                sum_all_words_logL = 0.0

                for word in self.hwords:
                    X, lengths = self.hwords[word]
                    sum_all_words_logL += model.score(X, lengths)

                sum_other_words_logL = sum_all_words_logL - current_word_logL
                dic_score = current_word_logL - sum_other_words_logL / (total_quantity_words - 1)
                if dic_score > max_dic_score:
                    max_dic_score = dic_score
                    best_model = model

            except Exception:
                return best_model

        return best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        DEFAULT_SPLITS = 3
        max_score = float('-inf')
        best_num_components = 0
        best_model = None

        # If the word sequence length is less than 2, KFold doesn't make sense
        if len(self.sequences) < 2:
            for num_states in range(self.min_n_components, self.max_n_components + 1):
                model = self.base_model(num_states)
                logL = model.score(self.X, self.lengths)
                if logL > max_score:
                    max_score = logL
                    best_model = model
            return best_model

        # Word sequence length could be 2 or greater
        for num_states in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = None
                sum_all_folds_logL = 0.0
                k_splits = min(DEFAULT_SPLITS, len(self.sequences))
                kfold = KFold(n_splits=k_splits)

                # Indexes of the folds, such as [2 3 4 5]
                for train_indexes, test_indexes in kfold.split(self.sequences):
                    # Fit the model on the training data
                    X_train, train_lengths = combine_sequences(train_indexes, self.sequences)
                    model = GaussianHMM(n_components=num_states,
                                        covariance_type='diag',
                                        n_iter=1000,
                                        random_state=self.random_state,
                                        verbose=False
                                        ).fit(X_train, train_lengths)
                    # Score the model on the test data
                    X_test, test_lengths = combine_sequences(test_indexes, self.sequences)
                    sum_all_folds_logL += model.score(X_test, test_lengths)

                # Current CV score is the average of log likelihoods across all folds
                cv_score = sum_all_folds_logL / k_splits
                if cv_score > max_score:
                    max_score = cv_score
                    best_num_components = num_states

            except Exception:
                return self.base_model(best_num_components)

        # Return a model fitted over all the data with the best number of hidden states
        return self.base_model(best_num_components)
