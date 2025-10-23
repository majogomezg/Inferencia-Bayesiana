Construcción de una Red Bayesiana en Python con pgmpy

Este proyecto utiliza la librería pgmpy
 para modelar y realizar inferencias en redes Bayesianas discretas.
A continuación se resumen los pasos requeridos para construir, verificar e inferir sobre una red según la documentación oficial de la librería.

1. Definir la estructura del modelo

La estructura se representa mediante un grafo acíclico dirigido (DAG), donde cada arco indica una relación causal entre variables (nodos).

from pgmpy.models import DiscreteBayesianNetwork

# Definición del grafo
model = DiscreteBayesianNetwork([
    ('Alcohol', 'Riesgo'),
    ('Clima', 'Riesgo'),
    ('Hora', 'Velocidad'),
    ('Velocidad', 'Riesgo'),
    ('Riesgo', 'Accidente'),
    ('Accidente', 'Severidad')
])


Cada tupla (A, B) indica que A es padre de B.

2. Especificar los estados de las variables

Cada variable discreta debe tener un conjunto de estados o categorías posibles:

states = {
    "Alcohol": ["No", "Si"],
    "Clima": ["Seco", "Lluvia"],
    "Hora": ["Dia", "Noche"],
    "Velocidad": ["Normal", "Alta"],
    "Riesgo": ["Bajo", "Alto"],
    "Accidente": ["No", "Si"],
    "Severidad": ["Leve", "Grave"],
}

3. Definir las Tablas de Probabilidad Condicional (CPDs)

Las CPDs describen las dependencias probabilísticas de cada nodo con respecto a sus padres.

from pgmpy.factors.discrete import TabularCPD

# Variable sin padres
cpd_alcohol = TabularCPD("Alcohol", 2, [[0.8], [0.2]])

# Variable con padres
cpd_riesgo = TabularCPD(
    "Riesgo", 2,
    [[0.9, 0.7, 0.6, 0.3, 0.6, 0.3, 0.2, 0.1],
     [0.1, 0.3, 0.4, 0.7, 0.4, 0.7, 0.8, 0.9]],
    evidence=["Alcohol", "Clima", "Velocidad"],
    evidence_card=[2, 2, 2]
)


Cada columna representa una combinación posible de los estados de los padres.
El orden de las columnas sigue la convención en que el último evidence varía más rápido.

4. Agregar las CPDs al modelo

Después de definir las tablas, se agregan al modelo:

model.add_cpds(cpd_alcohol, cpd_riesgo, cpd_clima, cpd_hora, cpd_velocidad, cpd_accidente, cpd_severidad)

5. Verificar la consistencia del modelo

Antes de ejecutar inferencias, se valida que todas las CPDs sean consistentes y que las columnas sumen 1.

print("Modelo válido:", model.check_model())


Si el resultado es True, el modelo es consistente.

6. Aplicar inferencia Bayesiana

Para realizar consultas probabilísticas condicionadas, se usa el método de eliminación de variables.

from pgmpy.inference import VariableElimination

infer = VariableElimination(model)

# Ejemplo: calcular probabilidad de severidad dada evidencia
q = infer.query(variables=["Severidad"], evidence={"Alcohol": "Si", "Clima": "Lluvia"})
print(q)


El resultado devuelve la distribución posterior de la variable consultada (en este caso, Severidad) dadas las evidencias especificadas.

7. Interpretar los resultados

La salida de pgmpy muestra los valores de probabilidad para cada estado de la variable consultada.
Por ejemplo:

+-------------+-----------+
| Severidad   | phi(Severidad) |
+=============+===========+
| Leve        | 0.6234 |
| Grave       | 0.3766 |
+-------------+-----------+


Esto se interpreta como:

Dadas las condiciones Alcohol=Sí y Clima=Lluvia, la probabilidad de que el accidente sea grave es del 37.7%, y de que sea leve es del 62.3%.
