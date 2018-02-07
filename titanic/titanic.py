import pandas
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from utils.misc import EmptyObject
from utils.filestuff import source_path

pandas.set_option('display.width', 1000)

CROSS_VALIDATION_SPLITS = 100
CROSS_VALIDATION_TEST_SIZE = 0.1

SHUFFLES = 100

doc_path = os.path.join(source_path(__file__), "doc")
submission_path = os.path.join(source_path(__file__), "submissions")


def age_categorisation(df):
    """
    Sourced from https://en.wikipedia.org/wiki/Human_development_(biology)
    :return:
    """

    df['Age_Infant'] = [1 if 0 <= age < 1 else 0 for age in df['Age']]
    df['Age_Toddler'] = [1 if 1 <= age < 3 else 0 for age in df['Age']]
    df['Age_Preschooler'] = [1 if 3 <= age < 5 else 0 for age in df['Age']]
    df['Age_Child'] = [1 if 5 <= age < 10 else 0 for age in df['Age']]
    df['Age_Preteen'] = [1 if 10 <= age < 13 else 0 for age in df['Age']]
    df['Age_Teenager'] = [1 if 13 <= age < 20 else 0 for age in df['Age']]
    df['Age_Adult'] = [1 if 20 <= age < 40 else 0 for age in df['Age']]
    df['Age_Midlife'] = [1 if 40 <= age < 60 else 0 for age in df['Age']]
    df['Age_Senior'] = [1 if 60 <= age < 1000 else 0 for age in df['Age']]
    df['Age_Unknown'] = [0 if age is None or 0 <= age < 1000 else 1 for age in df['Age']]
    df.drop(['Age'], axis=1)
    return df


def port_categorisation(df):
    # Embarkation port
    df['Port_Cherbourg'] = [1 if port == "C" else 0 for port in df['Embarked']]
    df['Port_Queenstown'] = [1 if port == "Q" else 0 for port in df['Embarked']]
    df['Port_Southampton'] = [1 if port == "S" else 0 for port in df['Embarked']]
    df['Port_Unknown'] = [1 if port not in ["C", "S", "Q"] else 0 for port in df['Embarked']]
    df = df.drop(['Embarked'], axis=1)
    return df


def sex_categorisation(df):
    # Sex
    df['Sex_Male'] = [1 if sex == "male" else 0 for sex in df['Sex']]
    df['Sex_Female'] = [1 if sex == "female" else 0 for sex in df['Sex']]
    # No unknown genders on record
    df = df.drop(['Sex'], axis=1)
    return df


def pre_processing(df):
    # Let's perform dummy encoding for our categorical data
    # df = age_categorisation(df)
    # df = port_categorisation(df)
    df = sex_categorisation(df)

    # drop unused metrics (for now at least)
    df = df.drop(['PassengerId', 'Pclass', 'Name', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],
                 axis=1)
    return df


def train():
    best_classifier = EmptyObject()
    best_classifier.f1 = 0
    best_classifier.precision = 0
    best_classifier.recall = 0
    best_classifier.accuracy = 0

    precision_sum = recall_sum = accuracy_sum = f1_sum = 0

    training_data_file = os.path.join(doc_path, "train.csv")
    training_df = pre_processing(pandas.read_csv(training_data_file))

    # TODO - stratified cross validation to go here ...
    # TODO - ... followed by a proper pipelining

    for _ in range(0,SHUFFLES):
        x_training, \
        x_validation, \
        y_training, \
        y_validation = train_test_split(training_df.drop(['Survived'], axis=1),
                                        training_df.Survived,
                                        test_size=CROSS_VALIDATION_TEST_SIZE)

        '''print(training_features)
        print(validation_features)
        print(training_survival)
        print(validation_survival)'''

        linear_svm_classifier = SVC(kernel='linear').fit(x_training, y_training)
        linear_svm_classifier_predictions = linear_svm_classifier.predict(x_validation)
        '''print(linear_svm_classifier.score(training_features, training_survival),
              linear_svm_classifier.score(validation_features, validation_survival))'''

        precision = precision_score(y_validation, linear_svm_classifier_predictions)
        recall = recall_score(y_validation, linear_svm_classifier_predictions)
        accuracy = accuracy_score(y_validation, linear_svm_classifier_predictions)
        f1 = f1_score(y_validation, linear_svm_classifier_predictions)

        # No penalty in false positives or false negatives in our data set, so just use f1 as our indicating metric

        precision_sum += precision
        recall_sum += recall
        accuracy_sum += accuracy
        f1_sum += f1

        if best_classifier.f1 < f1:
            best_classifier = linear_svm_classifier
            best_classifier.f1 = f1
            best_classifier.precision = precision
            best_classifier.recall = recall
            best_classifier.accuracy = accuracy

    f1_avg = f1_sum/SHUFFLES
    precision_avg = precision_sum / SHUFFLES
    recall_avg = recall_sum / SHUFFLES
    accuracy_avg = accuracy_sum / SHUFFLES

    return best_classifier, accuracy_avg, precision_avg, recall_avg, f1_avg


def evaluate(classifier):
    test_data_file = os.path.join(doc_path, "test.csv")
    test_df = pandas.read_csv(test_data_file)
    processed_df = pre_processing(test_df)

    predictions = classifier.predict(processed_df)
    #print(predictions)

    with open(os.path.join(submission_path, "submit002.csv"), 'w') as f:
        f.write("PassengerId,Survived\n")
        for i in range(0, len(predictions)):
            f.write("{0},{1}\n".format(test_df[['PassengerId']].values[i][0], predictions[i]))


if __name__ == "__main__":
    best_classifier, accuracy, precision, recall, f1 = train()
    print("Metric", "Avg", "Best")
    print("Precision", precision, best_classifier.precision)
    print("Recall", recall, best_classifier.recall)
    print("Accuracy", accuracy, best_classifier.accuracy)
    print("f1", f1, best_classifier.f1)

    evaluate(best_classifier)
