import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# When want to predict specific tweete, start by calling this method.
# json_file is the file path to the json file.
def run_tree(clf, json_file):
    # Extracts information from json file.
    # SUBNOTE: While the code will most likely not go here, it's important to make sure to delete those json files at some point to reduce the amount of 
    # space that the application takes up.
    testing_X = get_test_tweet(json_file)
    # Prints out the information.
    print_info(clf, testing_X)

# Gets the information needed for the test tweet.
def get_test_tweet(json_file):
    tweet_json = open(json_file, 'r', encoding='utf8')
    data = json.load(tweet_json)
    tweeter = data['includes']['users'][0] # This assumes that the first user in the list is the person who tweeted it.
                                           # If not, change it to the correct value.
    verified = []
    followers_count = []
    following_count = []
    tweet_count = [] 
    listed_count = []

    #Gets the right information, and puts it into a series of lists, which will then be converted into a dataframe.
    verified.append(tweeter['verified'])
    followers_count.append(tweeter['public_metrics']['followers_count'])
    following_count.append(tweeter['public_metrics']['following_count'])
    tweet_count.append(tweeter['public_metrics']['tweet_count'])
    listed_count.append(tweeter['public_metrics']['listed_count'])

    # Turns info into a dataframe, which is then returned.
    testing_X = pd.DataFrame(verified, columns=['Verified'])
    testing_X['Followers Count'] = followers_count
    testing_X['Following Count'] = following_count
    testing_X['Tweet Count'] = tweet_count
    testing_X['Listed Count'] = listed_count

    return testing_X

# Prints out the info from the decision tree.
def print_info(clf, testing_X):
    verified = ["Verified", None]
    followers = ["Followers Count", None, None]
    following = ["Following Count", None, None]
    tweet = ["Tweet Count", None, None]
    listed = ["Listed Count", None, None]
    nice_format = [verified, followers, following, tweet, listed]

    # Gets the info needed to obtain branches taken.
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold
    node_indicator = clf.decision_path(testing_X)
    leaf_id = clf.apply(testing_X)

    sample_id = 0
    # obtain ids of the nodes `sample_id` goes through, i.e., row `sample_id`
    node_index = node_indicator.indices[
                 node_indicator.indptr[sample_id]: node_indicator.indptr[sample_id + 1]
                 ]
    # Prints out tree branches taken.
    for node_id in node_index:
        # continue to the next node if it is a leaf node
        if leaf_id[sample_id] == node_id:
            continue
        if testing_X.iloc[sample_id, feature[node_id]] <= threshold[node_id]:
            threshold_sign = "<="
        else:
            threshold_sign = ">"

        # This can be commented or uncommented, it prints out the path the decision tree takes.
        # node: the branch that was taken
        # column: the variable that the branch is corresponding to (Verfied, Tweet Count, etc)
        # value: the value of said variable
        # inequality: if value was greater than or less than the threshold.
        # threshold: if value is greater than threshold, take right branch, else take left branch.
        print(
            "decision node {node} : ({column} = {value}) "
            "{inequality} {threshold})".format(
                node=node_id,
                column=testing_X.columns[feature[node_id]],
                value=testing_X.iloc[sample_id, feature[node_id]],
                inequality=threshold_sign,
                threshold=threshold[node_id],
            )
        )
        # To here.

        # Code to get nicer output.
        if feature[node_id] == 0:
            if threshold_sign == "<=":
                nice_format[feature[node_id]][1] = "False"
            else:
                nice_format[feature[node_id]][1] = "True"
        else:
            if threshold_sign == "<=":
                if nice_format[feature[node_id]][2] is None:
                    nice_format[feature[node_id]][2] = threshold[node_id]
                elif nice_format[feature[node_id]][2] > threshold[node_id]:
                    nice_format[feature[node_id]][2] = threshold[node_id]
            else:
                if nice_format[feature[node_id]][1] is None:
                    nice_format[feature[node_id]][1] = threshold[node_id]
                elif nice_format[feature[node_id]][1] < threshold[node_id]:
                    nice_format[feature[node_id]][1] = threshold[node_id]
    print("\n")

    # Prints out the summarized version of the information.
    # feature is the variable currently being printed.
    for feature in nice_format:
        if feature[0] == 'Verified':
            print(feature[0], ":", feature[1])
        else:
            # if no less than case when taking branches.
            if feature[1] is None:
                print(feature[0], "<=", feature[2])
            # if no greater than case when taking branches.
            elif feature[2] is None:
                print(feature[1], "<", feature[0])
            # if both greater than and less than cases when taking branches.
            else:
                print(feature[1], "<", feature[0], "<=", feature[2])
    print("\n")

    # Prints out final prediction.
    # A prediction of 0 means "false", or in this case, that the user's information suggests that they will post misinformation.
    # A prediction of 1 means "true", or that the user is more likely to tweet trustworhty information.
    if clf.predict(testing_X) == 0:
        print("Prediction: User untrustworthy")
    else:
        print("Prediction: User trustworthy")

# Change the name of this function as needed. It initializes the decision tree.
def main():
    # First run through of decision tree.
    full_df = pd.read_csv("all_data.csv") # Make sure path to all_data.csv is correct.
    X = full_df.iloc[:, [3, 5, 6, 7, 8]]
    y = full_df.loc[:, 'Misinformation']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=0)

    clf = DecisionTreeClassifier(random_state = 0)
    clf.fit(X_train, y_train)
    
    run_tree(clf, "template.json")

if __name__ == '__main__':
    main()
