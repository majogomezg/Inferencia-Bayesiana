from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# 1. Crear estructura
model = DiscreteBayesianNetwork([
    ('Lluvia','Mantenimiento'),
    ('Lluvia','Tren'),
    ('Mantenimiento','Tren'),
    ('Tren','Cita')
])

# 2. Crear tablas (CPDs)


# CPD para Lluvia
cpd_lluvia = TabularCPD('Lluvia', 2, [[0.7], [0.3]])  # No, Sí
# CPD para Mantenimiento
cpd_mantenimiento = TabularCPD('Mantenimiento', 2, [[0.9, 0.2], [0.1, 0.8]], evidence=['Lluvia'], evidence_card=[2])
# CPD para Tren: depende de Lluvia y Mantenimiento
cpd_tren = TabularCPD(
    variable='Tren',
    variable_card=2,
    values=[[0.95, 0.8, 0.7, 0.1], [0.05, 0.2, 0.3, 0.9]],
    evidence=['Lluvia', 'Mantenimiento'],
    evidence_card=[2, 2]
)
# CPD para Cita: depende de Tren
cpd_cita = TabularCPD(
    variable='Cita',
    variable_card=2,
    values=[[0.9, 0.6], [0.1, 0.4]],
    evidence=['Tren'],
    evidence_card=[2]
)

model.add_cpds(cpd_lluvia, cpd_mantenimiento, cpd_tren, cpd_cita)
model.check_model()

# 3. Consultar
inference = VariableElimination(model)
resultado = inference.query(variables=['Cita'], evidence={'Lluvia':1, 'Mantenimiento':0})
print(resultado)
