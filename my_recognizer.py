import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key is a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    probabilities = []
    guesses = []

    # For each test word (string unknown), the recognizer will determine the answer to:
    #     What word do these test word sequences most likely represent?
    for word_idx in range(0, len(test_set.get_all_Xlengths())):
        sequences = test_set.get_item_sequences(word_idx)
        X_test, test_lengths = test_set.get_item_Xlengths(word_idx)
        max_score = float('-inf')
        best_guess_word = None
        test_word_probabilities = {}

        # `models` is a dict of words, in which each word is mapped to the
        #     best model (either CV, BIC, or DIC -- same type for all words in `models`)
        for word in models:
            model = models[word]
            logL = float('-inf')

            if model is not None:
                try:
                    # Calculate the scores for each word model (trained on the word's
                    #     training data) using the test data
                    logL = model.score(X_test, test_lengths)
                except Exception:
                    logL = float('-inf')

            # If there is no model, save the test word probabilities as:
            #     { 'FISH': float('-inf') }
            test_word_probabilities[word] = logL

            # Keep track of the best score (highest log likelihood) and the most likely word
            if logL > max_score:
                max_score = logL
                best_guess_word = word

        # Save the log likelihood of each word in models against the current test word index
        probabilities.append(test_word_probabilities)
        # Save the best guess word string for the current test word index
        guesses.append(best_guess_word)

    return probabilities, guesses
