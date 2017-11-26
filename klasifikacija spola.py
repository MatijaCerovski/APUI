import nltk
import random
from nltk.corpus import names


def gender_features(word):
    #return {'last_letter': word[-1]}
    return {'suffix1': word[-1:],
            'suffix2': word[-2:]}


def gender_features2(name):
    features = {}
    features["first_letter"] = name[0].lower()
    features["last_letter"] = name[-1].lower()
    for letter in 'abcdefghijklmnopqrstuvwxyz':
        features["count({})".format(letter)] = name.lower().count(letter)
        features["has({})".format(letter)] = (letter in name.lower())
    return features


def main():
    #Dodavanje imana
    labeled_names = ([(name, 'male') for name in names.words('male.txt')] +
                     [(name, 'female') for name in names.words('female.txt')])
    random.shuffle(labeled_names)

    #Raspodjela imena 1
    featuresets = [(gender_features(n), gender) for (n, gender) in labeled_names]
    train_set, test_set = featuresets[500:], featuresets[:500]

    #Raspodjela imena 2
    train_names = labeled_names[1500:]
    devtest_names = labeled_names[500:1500]
    test_names = labeled_names[:500]
    train_set = [(gender_features(n), gender) for (n, gender) in train_names]
    devtest_set = [(gender_features(n), gender) for (n, gender) in devtest_names]
    test_set = [(gender_features(n), gender) for (n, gender) in test_names]


    #Treniranje klasifikatora
    classifier = nltk.NaiveBayesClassifier.train(train_set)

    #Ispis Rezultata
    print(gender_features2("John"))
    print(classifier.classify(gender_features2('Neo')))
    print(classifier.classify(gender_features2('Trinity')))
    print(nltk.classify.accuracy(classifier, devtest_set))
    print(classifier.show_most_informative_features(5))

    #Ispis krivih pretpostavki
    errors = []
    for (name, tag) in devtest_names:
        guess = classifier.classify(gender_features(name))
        if guess != tag:
            errors.append((tag, guess, name))

    for (tag, guess, name) in sorted(errors):
        print('correct={:<8} guess={:<8s} name={:<30}'.format(tag, guess, name))

if __name__ == "__main__": main()
