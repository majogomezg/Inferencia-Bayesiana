Definir la estructura (DAG)

Crea el modelo y añade arcos dirigidos padre → hijo. En pgmpy para redes discretas se usa DiscreteBayesianNetwork (el API es análogo si usas BayesianNetwork). 
pgmpy
+1

from pgmpy.models import DiscreteBayesianNetwork
model = DiscreteBayesianNetwork([
    ('A', 'B'),
    ('C', 'B'),
    # ...
])


Especificar las CPDs (Tablas de Probabilidad Condicional)

Cada nodo debe tener una CPD: tablas de una columna para nodos sin padres y tablas con columnas indexadas por las combinaciones de los padres para nodos con padres. La clase a usar es TabularCPD. 
pgmpy
+1

from pgmpy.factors.discrete import TabularCPD

# Nodo sin padres:
cpd_A = TabularCPD("A", 2, [[0.7], [0.3]])

# Nodo con padres (el último evidence varía más rápido en 'values'):
cpd_B = TabularCPD(
    "B", 2,
    [[0.9, 0.2, 0.6, 0.1],   # B=estado0
     [0.1, 0.8, 0.4, 0.9]],  # B=estado1
    evidence=["A", "C"],
    evidence_card=[2, 2]
)


La documentación muestra cómo definir CPDs (incluida la variante con state_names) y proporciona utilidades como get_uniform/get_random para generar tablas válidas. 
pgmpy
+1

Asociar todas las CPDs al modelo

Una vez definidas, deben agregarse al modelo con add_cpds. 
pgmpy

model.add_cpds(cpd_A, cpd_B, /* ... resto de CPDs ... */)


Verificar la consistencia del modelo

Usa check_model() para comprobar: (i) que todas las CPDs requeridas están presentes, (ii) que las dimensiones concuerdan con los padres, y (iii) que cada columna de la CPD suma 1 (tolerancia típica 0.01). Devuelve True si todo es consistente. 
pgmpy
+1

assert model.check_model()


Inicializar el motor de inferencia

pgmpy implementa inferencias exactas como VariableElimination (y también BeliefPropagation con API equivalente). 
pgmpy

from pgmpy.inference import VariableElimination
infer = VariableElimination(model)


Formular consultas con evidencia

Para obtener 
𝑃
(
𝑋
∣
𝐸
=
𝑒
)
P(X∣E=e), llama infer.query(variables=[...], evidence={...}). El algoritmo realiza la eliminación de variables y devuelve una distribución sobre la(s) variable(s) consultada(s). La documentación de VariableElimination describe el método y utilidades relacionadas. 
pgmpy
+1

q = infer.query(variables=["B"], evidence={"A": "estado0", "C": "estado1"})
print(q)            # tabla con P(B | A, C)
print(q.values)     # numpy array con los valores
