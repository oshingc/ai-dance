import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import numpy as np

# Cargar datos
data = pd.read_csv("reference_keypoints_misamo.csv")

# Crear etiquetas (aquí necesitas etiquetar tus datos como 'correcto' o 'incorrecto')
# Este es un paso crítico. Necesitas crear una columna 'label' que indique si el movimiento es correcto o incorrecto.
# Puedes hacerlo manualmente o usando un script para generar etiquetas basadas en tus criterios.
# Para simplificar, asumiremos que todos los movimientos son correctos (label=1)
data['label'] = 1

# Dividir datos en features y labels
X = data.drop('label', axis=1)
y = data['label']

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir el modelo
model = RandomForestClassifier(random_state=42)

# Definir el espacio de búsqueda de hiperparámetros
param_dist = {
    'n_estimators': np.arange(50, 200, 10),
    'max_depth': np.arange(3, 15, 2),
    'min_samples_split': np.arange(2, 10, 2),
    'min_samples_leaf': np.arange(1, 5, 1)
}

# Configurar RandomizedSearchCV
random_search = RandomizedSearchCV(
    model,
    param_distributions=param_dist,
    n_iter=10,  # Número de combinaciones a probar
    cv=3,  # Número de folds para validación cruzada
    verbose=2,
    random_state=42,
    n_jobs=-1  # Usar todos los núcleos de la CPU
)

# Entrenar el modelo
random_search.fit(X_train, y_train)

# Mostrar los mejores hiperparámetros
print("Mejores hiperparámetros:", random_search.best_params_)

# Evaluar el modelo
y_pred = random_search.predict(X_test)
print(classification_report(y_test, y_pred))