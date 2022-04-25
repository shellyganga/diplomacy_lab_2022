import pandas as pd

from ttp import ttp
p = ttp.Parser()
df = pd.read_csv("/Users/shellyschwartz/Downloads/webis-clickbait-16/truth/data2.csv")

df = df.iloc[: , 6:]
df = df.dropna()


def get_text(text):

    text_fin = ""

    result = p.parse(text)


    arr = text.split(" ")

    for elem in arr:
        if ((elem not in str(result.users)) and (elem not in str(result.tags)) and (elem not in str(result.urls))):
            text_fin = text_fin  + elem + " "

    return text_fin[:-1]

for idx, row in df.iterrows():
    df.loc[idx, 'text'] = get_text(df.loc[idx, 'text'])


import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

# Manually fill out to map text labels to integers
LABEL_MAP = {}
LABELS_LIST = []

from imblearn.over_sampling import SMOTE

def load_data():
    """
    Loads data from csv file.

    Returns:
        A list of descriptions (np.array(features)) and a list of corresponding labels
    """

    # np.array(features) = []
    # labels = []
    # with open('data.csv', 'r') as infile:
    #     reader = csv.reader(infile)
    #     next(reader)
    #     for row in reader:
    #         np.array(features).append(row[0])
    #         labels.append(LABEL_MAP[row[1]])

    return np.array(df['text']), np.array(df['clickbait'])


def get_subset(dataSet, indexes):
    """
    Selects subset of data based on indexes given
    from KFold validation method
    Returns:
        Subset of data set
    """
    return [dataSet[i] for i in indexes]


def make_svc_classifier(features_train, features_test, labels_train):
    """
    Creates SVC Classifier
    Uses TFIDF to generate features for SVC. See Wikipedia for definition,
    but basically a normalized way to calculate frequency of words in documents.
    Args:
        features_train: List of training data
        fetures_test: List of test data
        labels_train: List of classifications for training data
    Returns:
        SVC classifier returned
    """
    clf = Pipeline([
        ('vect', TfidfVectorizer(sublinear_tf=True, max_df=.5, stop_words='english')),
        ('selector', SelectPercentile(f_classif, percentile=85)),
        ('clf', SVC(kernel='rbf', C=6000, gamma=.0002))
    ])
    return clf


def make_logistic_regression_classifier(features_train, features_test, labels_train):
    """
    Creates Logistic Regression Classifier
    Also uses TFIDF to generate features.
    Args:
        features_train: List of training data
        fetures_test: List of test data
        labels_train: List of classifications for training data
    Returns:
        Logistic regression classifier returned
    """
    clf = Pipeline([
        ('vect', TfidfVectorizer(sublinear_tf=True, max_df=.95, min_df=.001, stop_words='english')),
        ('selector', SelectPercentile(f_classif, percentile=100)),
        ('clf', LogisticRegression(C=100))
    ])
    return clf


def make_nb_classifier(features_train, features_test, labels_train):
    """
    Creates Multinomial Naive Bayes Classifier
    Not the same Naive Bayes algorithm as in NLTK.
    Uses counts of words without normalization for features. Uses n-grams up to
    four words.
    Args:
        features_train: List of training data
        fetures_test: List of test data
        labels_train: List of classifications for training data
    Returns:
        Naive Bayes classifier returned
    """
    clf = Pipeline([('vect', CountVectorizer(stop_words='english', max_df=0.3, ngram_range=(1,4))),
        ('selector', SelectPercentile(f_classif, percentile=100)),
        ('clf', MultinomialNB())
    ])
    return clf

def run_algorithm(make_clf, features, labels):
    """
    Runs specified algorithm and prints metrics and accuracy
    Runs 10-Fold cross validation and returns average accuracy
    and metrics.
    The following metrics are returns:
    - precision
    - recall
    - f1 score
    - support
    Args:
        make_clf: Callback function to create classifier
        features: List of example data
        labels: List of corresponding labels for data
    """
    accuracy_scores = []
    metrics = []

    kf = StratifiedKFold(n_splits=5,shuffle=True,random_state=4)
    for train_index, test_index in kf.split(features, labels):
        features_train = get_subset(features, train_index)
        labels_train = get_subset(labels, train_index)
        features_test = get_subset(features, test_index)
        labels_test = get_subset(labels, test_index)

        clf = make_clf(features_train, features_test, labels_train)
        clf.fit(features_train, labels_train)
        pred = clf.predict(features_test)

        accuracy_scores.append(accuracy_score(labels_test, pred))
        metrics.append(precision_recall_fscore_support(labels_test, pred, average=None))

    print_metrics(accuracy_scores, metrics)

def run_random_forests(features, labels):
    """
    Special function to run Random Forests
    Couldn't figure out how to use Pipeline class with
    Random Forests due to requirement that Sparse matrices
    are not allowed.
    Basically does the same thing as run_algorithm except it
    only does the Random Forests algorithm.
    Args:
        features: List of example data
        labels: List of corresponding labels for data
    """
    accuracy_scores = []
    metrics = []

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=4)
    for train_index, test_index in kf.split(features, labels):

        # Get subsets of training and test data
        features_train =  features[train_index]

        labels_train = labels[test_index]



        features_test = get_subset(features, test_index)
        labels_test = get_subset(labels, test_index)

        # Create TFIDF Vectorizer to generate features
        vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=1.0, min_df=0.002, stop_words='english')
        features_train_transformed = vectorizer.fit_transform(features_train)
        features_test_transformed  = vectorizer.transform(features_test)

        # Select features in top X percentile. Here 100 seemed to work best
        selector = SelectPercentile(f_classif, percentile=100)
        selector.fit(features_train_transformed, labels_train)

        # Need to make sure non-sparse matrices are used
        features_train = selector.transform(features_train_transformed).toarray()
        features_test = selector.transform(features_test_transformed).toarray()

        # Create classifier, fit and predict
        clf = RandomForestClassifier(min_samples_split=15, criterion='gini')
        clf.fit(features_train, labels_train)
        pred = clf.predict(features_test)

        accuracy_scores.append(accuracy_score(labels_test, pred))
        metrics.append(precision_recall_fscore_support(labels_test, pred, average=None))

    print_metrics(accuracy_scores, metrics)


def print_metrics(accuracy_scores, metrics):
    """
    Prints metrics and accuracy for given data.
    Assumes 10-K Cross validation is used
    Args:
        accuracy_scores: List of scores from each run
        metrics: List of metrics from each run
    """
    precision_results = [0] * len(LABELS_LIST)
    recall_results = [0] * len(LABELS_LIST)
    f1_results = [0] * len(LABELS_LIST)
    support_results = 0

    for precision, recall, f1, support in metrics:
        for i, label in enumerate(LABELS_LIST):
            precision_results[i] += precision[i]
            recall_results[i] += recall[i]
            f1_results[i] += f1[i]
        support_results += support

    for i, label in enumerate(LABELS_LIST):
        output = [label]
        for v in (precision_results[i], recall_results[i], f1_results[i]):
            output.append("{0:0.{1}f}".format(v/10., 2))
        output.append("{0}".format(support_results[i]/10.))
        print ('\t'.join(output))

    print("Accuracy: {0:.2f}".format(np.array(accuracy_scores).mean()))


def main():
    features, labels = load_data()

    print("Random Forests Results:")
    run_random_forests(features, labels)

    print
    print("Multinomial Naive Bayes Results:")
    run_algorithm(make_nb_classifier, features, labels)




    # print
    # print("Logistic Regression Results:")
    # run_algorithm(make_logistic_regression_classifier, features, labels)
    #
    # print
    # print("SVC Results:")
    # run_algorithm(make_svc_classifier, features, labels)


if __name__ == '__main__':
    main()