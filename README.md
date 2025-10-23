Construcción de una Red Bayesiana con pgmpy

Este proyecto implementa una red Bayesiana en Python utilizando la librería pgmpy
.
Para usar de ejemplo se usa una red que modela la relación entre variables que influyen en la probabilidad de accidentes viales, integrando factores humanos y ambientales.
A continuación se describen los pasos requeridos para construir el modelo, verificar su consistencia y realizar inferencias probabilísticas.

1. Definir la estructura del modelo

El primer paso consiste en definir el grafo acíclico dirigido (DAG) que describe las dependencias entre las variables.
Cada arco indica una relación causal de la forma padre → hijo.

from pgmpy.models import DiscreteBayesianNetwork

`model = DiscreteBayesianNetwork([
    ('Alcohol', 'Riesgo'),
    ('Clima', 'Riesgo'),
    ('Hora', 'Velocidad'),
    ('Velocidad', 'Riesgo'),
    ('Riesgo', 'Accidente'),
    ('Accidente', 'Severidad')
])`


En este ejemplo, Alcohol y Clima influyen sobre Riesgo, y este a su vez afecta la probabilidad de Accidente, el cual determina la Severidad.

2. Definir los estados de las variables

Cada nodo de la red representa una variable discreta con estados definidos:

`states = {
    "Alcohol": ["No", "Si"],
    "Clima": ["Seco", "Lluvia"],
    "Hora": ["Dia", "Noche"],
    "Velocidad": ["Normal", "Alta"],
    "Riesgo": ["Bajo", "Alto"],
    "Accidente": ["No", "Si"],
    "Severidad": ["Leve", "Grave"],
}`

3. Definir las Tablas de Probabilidad Condicional (CPDs)

Cada variable debe asociarse con una tabla de probabilidad condicional (TabularCPD), que describe cómo varía su probabilidad según sus padres.

from pgmpy.factors.discrete import TabularCPD

`# Variable sin padres
cpd_alcohol = TabularCPD("Alcohol", 2, [[0.8], [0.2]])`

`# Variable con padres
cpd_riesgo = TabularCPD(
    "Riesgo", 2,
    [[0.9, 0.7, 0.6, 0.3, 0.6, 0.3, 0.2, 0.1],   # Riesgo = Bajo
     [0.1, 0.3, 0.4, 0.7, 0.4, 0.7, 0.8, 0.9]],  # Riesgo = Alto
    evidence=["Alcohol", "Clima", "Velocidad"],
    evidence_card=[2, 2, 2]
)`


Cada columna representa una combinación de los estados de las variables padre.
El último elemento de evidence es el que varía más rápido en la tabla.

4. Agregar las CPDs al modelo

Una vez creadas todas las tablas de probabilidad, se agregan al modelo:

`model.add_cpds(
    cpd_alcohol,
    cpd_clima,
    cpd_hora,
    cpd_velocidad,
    cpd_riesgo,
    cpd_accidente,
    cpd_severidad
)
`
5. Verificar la consistencia del modelo

Antes de ejecutar cualquier inferencia, se recomienda verificar que todas las CPDs sean coherentes con la estructura y que cada columna sume 1.

`print("Modelo válido:", model.check_model())`


Si el resultado es True, la red es consistente y puede utilizarse para inferencias.

6. Realizar inferencia Bayesiana

Para calcular probabilidades condicionadas se utiliza la clase VariableElimination, que aplica el algoritmo de eliminación de variables descrito en la documentación oficial.

`from pgmpy.inference import VariableElimination`

`infer = VariableElimination(model)
q = infer.query(variables=["Severidad"], evidence={"Alcohol": "Si", "Clima": "Lluvia"})
print(q)`

El método query devuelve un objeto que contiene los valores de probabilidad asociados a cada estado de la variable consultada.

7. Interpretación de los resultados

Para la consulta:

`P(Severidad | Alcohol=Si, Clima=Lluvia)`


El resultado podría ser:
`
Severidad = Leve  →  0.6234
Severidad = Grave →  0.3766
`

Esto significa que, dadas las condiciones de lluvia y consumo de alcohol, la red predice una probabilidad del 37.7% de que el accidente sea grave y del 62.3% de que sea leve.

8. Fuentes de referencia

Documentación oficial de pgmpy: https://pgmpy.org/

Módulo DiscreteBayesianNetwork: https://pgmpy.org/models/bayesiannetwork.html

Clases TabularCPD y VariableElimination:

https://pgmpy.org/factors/tabularcpd.html

https://pgmpy.org/inference/variable_elimination.html
