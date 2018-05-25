from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import pickle as pkl

if __name__ == "__main__":
    # load training set
    data_path = '/Users/liushuheng/Desktop/DecisionTreeData/data.csv'
    data = pd.read_csv(data_path)

    # get input features and ground-truth labels
    features = data.drop('label', axis=1)
    labels = data['label']

    # REVIEW consider other combinations of hyperparameters
    classifier = DecisionTreeClassifier(criterion='entropy', max_depth=4, min_samples_leaf=100, presort=True)

    # fit the classifier with feature-labels pairs
    classifier.fit(features, labels)
    score = classifier.score(features, labels)
    print(score)

    # dump the classifier
    dump_filename = "/Users/liushuheng/Desktop/DecisionTreeData/model%d.pkl" % (score * 1000)
    with open(dump_filename, 'wb') as fo:
        pkl.dump(classifier, fo)

    # score = 0.994
    # dump_filename = "/Users/liushuheng/Desktop/DecisionTreeData/model%d.pkl" % (score * 1000)
    # with open(dump_filename, 'rb') as f:
    #     new_classifier = pkl.load(f)  # type: DecisionTreeClassifier
    #
    # input = ((20000, 800000, 16000, 3.0, 0.91),)
    # prediction = new_classifier.predict(input)
