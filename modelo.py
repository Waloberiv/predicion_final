#importando libreria
import pandas as pd
import numpy as np

#cargamos los datos
print("Carga de datos")
glass_data = pd.read_csv("glass.csv")
#crearemos una variable target con valores numericos
glass_data['Type'] = np.where(glass_data['Type']=='building_windows_float_processed',1,
                             np.where(glass_data['Type']=='building_windows_non_float_processed',2,
                                     np.where(glass_data['Type']=='headlamps',3,
                                             np.where(glass_data['Type']=='vehicle_windows_float_processed',4,
                                                     np.where(glass_data['Type']=='containers',5,
                                                             np.where(glass_data['Type']=='tableware',6,
                                                        np.nan))))))

#limpieza de datos
print("Limpieza de datos")
q3, q1 = np.percentile(glass_data['K'], [75, 25])
iqr = q3 - q1
glass_data['K_outlier'] = np.where((glass_data['K']>q3+1.5*iqr) | (glass_data['K']<q1-1.5*iqr) ,1,0)
glass_data = glass_data[glass_data['K']<=2]

from sklearn.model_selection import train_test_split

#seleccionamos las variables
print("Datos para modelar")
X = glass_data[['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']]
y = glass_data['Type']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
#modelamiento

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import classification_report
from sklearn import metrics
print("Modelando")
models = {}
models['knn'] = KNeighborsClassifier()                #### K-neighbors
models['cart'] = DecisionTreeClassifier()             #### Decision Tree
models['svm'] = SVC()                                 #### Support vector machine
models['bayes'] = GaussianNB()                        #### Naive Bayes
models['rdm'] = RandomForestClassifier()              #### Random Forest
models['lgc'] = LogisticRegression(max_iter=1000)     #### Logistic Reggresion
models['ada'] = AdaBoostClassifier()                  #### Adaboost
models['gda'] = GradientBoostingClassifier()          #### Gradient Boosting
models['bca'] = BaggingClassifier()                   #### Bagging
print("Modelando : entrenamiento")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred = model.predict(X_test)
    accuracy_train = accuracy_score(y_train,y_pred_train)
    accuracy = accuracy_score(y_test,y_pred)
    #auc_score = metrics.roc_auc_score( y_test, y_pred,  )
    classification = classification_report(y_test,y_pred)
    print(name, 'Accuracy_train',accuracy_train,'Accuracy_test',accuracy)
    print(name, classification)

    # Prediccion
    y_pred = model.predict(X_test)
    # Reporte
    from sklearn.metrics import classification_report

    print(classification_report(y_test, y_pred))
    # List Hyperparameters that we want to tune.
    n_estimators = list(range(30, 35))
    max_features = list(range(5, 10))
    max_samples = list(range(30, 35))
    # Convert to dictionary
    hyperparameters = dict(n_estimators=n_estimators, max_features=max_features, max_samples=max_samples)
    # Create new KNN object
    bca = BaggingClassifier()
    # Use GridSearch
    from sklearn.model_selection import GridSearchCV

    clf = GridSearchCV(bca, hyperparameters, cv=10)
    # Fit the model
    best_model = clf.fit(X_train, y_train)

#Print The value of best Hyperparameters
print('Best n_estimators:', best_model.best_estimator_.get_params()['n_estimators'])
print('Best max_feature:', best_model.best_estimator_.get_params()['max_features'])
print('Best max_samples:', best_model.best_estimator_.get_params()['max_samples'])

model = BaggingClassifier(n_estimators=33,max_features=9,max_samples=31)
model.fit(X_train, y_train)
#Prediccion
y_pred = model.predict(X_test)
#Reporte
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred ))

#guardando el modelo final
from pickle import dump
from pickle import load
def guardar_modelo(model,nombre):
    output = open(nombre, 'wb')
    dump(model, output , -1)
    output.close()

#cargar modelo
def cargar_modelo(nombre):
    input = open(nombre, 'rb')
    modelo = load(input)
    input.close()
    return modelo

nombre = 'modelo.pkl'
guardar_modelo(model,nombre)