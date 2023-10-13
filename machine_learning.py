from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def lin_reg(data_five):
    y = data_five['Preis']
    X = data_five[['Kilometerstand', 'PS', 'Jahr']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=100)

    model = LinearRegression()
    model.fit(X_train, y_train)

    return X, X_train, X_test, y_train, y_test, model