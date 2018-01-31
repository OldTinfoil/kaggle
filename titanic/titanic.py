import pandas
import os
from utils.filestuff import source_path

pandas.set_option('display.width', 1000)


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
    df.drop(['Age'], 1)
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
    df['Sex_Male'] = [1 if sex == "male" else 0 for sex in training_df['Sex']]
    df['Sex_Female'] = [1 if sex == "female" else 0 for sex in training_df['Sex']]
    # No unknown genders on record
    df = df.drop(['Sex'], axis=1)
    return df


def pre_process_data(df):
    # Let's perform dummy encoding for our categorical data
    df = age_categorisation(df)
    df = port_categorisation(df)
    df = sex_categorisation(df)

    # drop unused metrics (for now at least

    return df


if __name__ == "__main__":
    doc_path = os.path.join(source_path(__file__), "doc")
    training_data_file = os.path.join(doc_path, "train.csv")
    training_df = pandas.read_csv(training_data_file)

    print(training_df.columns.values)
    # print(training_data.describe())
    # print(training_data.describe(include=['O']))

    training_df = pre_process_data(training_df)

    print(training_df.describe())

    print(training_df[['Age_Infant', 'Survived']].groupby(['Age_Infant'], as_index=False).mean().sort_values(
        by='Survived',
        ascending=False))

    print(training_df[['Age_Toddler', 'Survived']].groupby(['Age_Toddler'], as_index=False).mean().sort_values(
        by='Survived',
        ascending=False))

    print(training_df[['Age_Preschooler', 'Survived']].groupby(['Age_Preschooler'], as_index=False).mean().sort_values(
        by='Survived',
        ascending=False))

    print(
        training_df[['Age_Child', 'Survived']].groupby(['Age_Child'], as_index=False).mean().sort_values(by='Survived',
                                                                                                         ascending=False))

    print(training_df[['Age_Preteen', 'Survived']].groupby(['Age_Preteen'], as_index=False).mean().sort_values(
        by='Survived',
        ascending=False))

    print(training_df[['Age_Teenager', 'Survived']].groupby(['Age_Teenager'], as_index=False).mean().sort_values(
        by='Survived',
        ascending=False))

    print(
        training_df[['Age_Adult', 'Survived']].groupby(['Age_Adult'], as_index=False).mean().sort_values(by='Survived',
                                                                                                         ascending=False))

    print(training_df[['Age_Midlife', 'Survived']].groupby(['Age_Midlife'], as_index=False).mean().sort_values(
        by='Survived',
        ascending=False))

    print(training_df[['Age_Senior', 'Survived']].groupby(['Age_Senior'], as_index=False).mean().sort_values(
        by='Survived',
        ascending=False))

    print(training_df[['Age_Unknown', 'Survived']].groupby(['Age_Unknown'], as_index=False).mean().sort_values(
        by='Survived',
        ascending=False))

    print(training_df[['Age_Unknown', 'Survived']]).mean().sort_values(by='Survived', ascending=False)

    '''print(training_df.head())
    print(training_df.tail())

    print(training_df[['Port_Cherbourg', 'Survived']].groupby(['Port_Cherbourg'],
                                                              as_index=False).mean().sort_values(by='Survived',
                                                                                                 ascending=False))

    print(training_df[['Port_Queenstown', 'Survived']].groupby(['Port_Queenstown'],
                                                               as_index=False).mean().sort_values(by='Survived',
                                                                                                 ascending=False))

    print(training_df[['Port_Southampton', 'Survived']].groupby(['Port_Southampton'],
                                                                as_index=False).mean().sort_values(by='Survived',
                                                                                                 ascending=False))

    print(training_df[['Port_Unknown', 'Survived']].groupby(['Port_Unknown'],
                                                            as_index=False).mean().sort_values(by='Survived',
                                                                                                 ascending=False))

    print(training_df[['Sex_Male', 'Survived']].groupby(['Sex_Male'],
                                                        as_index=False).mean().sort_values(by='Survived',
                                                                                                 ascending=False))

    print(training_df[['Sex_Female', 'Survived']].groupby(['Sex_Female'],
                                                          as_index=False).mean().sort_values(by='Survived',
                                                                                                 ascending=False))

    print(training_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived',
                                                                                                         ascending=False))'''
