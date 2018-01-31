import pandas
import os
import matplotlib.pyplot as plt
import seaborn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils.filestuff import source_path
from sklearn.svm import SVC

pandas.set_option('display.width', 1000)

CROSS_VALIDATION_SPLITS = 100
CROSS_VALIDATION_TEST_SIZE = 0.1


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


def main():
    doc_path = os.path.join(source_path(__file__), "doc")
    training_data_file = os.path.join(doc_path, "train.csv")
    training_df = pandas.read_csv(training_data_file)

    test_data_file = os.path.join(doc_path, "test.csv")
    test_df = pandas.read_csv(test_data_file)

    training_df = pre_processing(training_df)
    test_df = pre_processing(test_df)

    # TODO - stratified cross validation to go here ...
    # TODO - ... followed by a proper pipelining

    training_features, \
    validation_features, \
    training_survival, \
    validation_survival = train_test_split(training_df.drop(['Survived'], axis=1),
                                           training_df.Survived,
                                           test_size=CROSS_VALIDATION_TEST_SIZE)

    '''print(training_features)
    print(validation_features)
    print(training_survival)
    print(validation_survival)'''

    linear_svm_classifier = SVC(kernel='linear').fit(training_features, training_survival)
    linear_svm_classifier_predictions = linear_svm_classifier.predict(validation_features)
    print(linear_svm_classifier.score(training_features, training_survival),
          linear_svm_classifier.score(validation_features, validation_survival))

    return linear_svm_classifier.score(validation_features, validation_survival)

if __name__ == "__main__":
    main()
