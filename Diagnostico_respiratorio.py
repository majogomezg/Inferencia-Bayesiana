from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# --- 1. Definición del grafo ---
model = BayesianNetwork([
    ('Virus', 'Fiebre'),
    ('Virus', 'Tos'),
    ('Virus', 'Congestión'),
    ('Alergia', 'Congestión'),
    ('Tabaquismo', 'Tos'),
    ('Fiebre', 'Saturación'),
    ('Tos', 'Diagnóstico'),
    ('Congestión', 'Diagnóstico'),
    ('Saturación', 'Diagnóstico')
])

# --- 2. Definición de las CPDs ---
cpd_virus = TabularCPD('Virus', 2, [[0.7], [0.3]])
cpd_alergia = TabularCPD('Alergia', 2, [[0.8], [0.2]])
cpd_tabaquismo = TabularCPD('Tabaquismo', 2, [[0.75], [0.25]])

cpd_fiebre = TabularCPD('Fiebre', 2, [[0.8, 0.2], [0.2, 0.8]], evidence=['Virus'], evidence_card=[2])
cpd_tos = TabularCPD('Tos', 2, 
    [[0.8, 0.6, 0.2, 0.1],
     [0.2, 0.4, 0.8, 0.9]],
    evidence=['Virus','Tabaquismo'], evidence_card=[2,2])

cpd_congestion = TabularCPD('Congestión', 2,
    [[0.7,0.5,0.2,0.1],
     [0.3,0.5,0.8,0.9]],
    evidence=['Virus','Alergia'], evidence_card=[2,2])

cpd_saturacion = TabularCPD('Saturación', 2, [[0.7,0.1],[0.3,0.9]], evidence=['Fiebre'], evidence_card=[2])

cpd_diagnostico = TabularCPD('Diagnóstico', 2,
    [[0.95,0.8,0.7,0.6,0.4,0.3,0.2,0.05],
     [0.05,0.2,0.3,0.4,0.6,0.7,0.8,0.95]],
    evidence=['Tos','Congestión','Saturación'], evidence_card=[2,2,2])

model.add_cpds(cpd_virus, cpd_alergia, cpd_tabaquismo, cpd_fiebre, cpd_tos, cpd_congestion, cpd_saturacion, cpd_diagnostico)

# --- 3. Inferencias ---
infer = VariableElimination(model)

print("\nP(Diagnóstico | Fiebre=Alta, Tos=Sí):")
print(infer.query(variables=['Diagnóstico'], evidence={'Fiebre':1, 'Tos':1}))

print("\nP(Diagnóstico | Congestión=Sí, Saturación=Baja):")
print(infer.query(variables=['Diagnóstico'], evidence={'Congestión':1, 'Saturación':0}))

print("\nP(Tos | Virus=Sí):")
print(infer.query(variables=['Tos'], evidence={'Virus':1}))

print("\nP(Congestión | Alergia=Sí):")
print(infer.query(variables=['Congestión'], evidence={'Alergia':1}))

print("\nP(Saturación | Fiebre=Alta):")
print(infer.query(variables=['Saturación'], evidence={'Fiebre':0}))
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# --- 1. Definición del grafo ---
model = BayesianNetwork([
    ('Virus', 'Fiebre'),
    ('Virus', 'Tos'),
    ('Virus', 'Congestión'),
    ('Alergia', 'Congestión'),
    ('Tabaquismo', 'Tos'),
    ('Fiebre', 'Saturación'),
    ('Tos', 'Diagnóstico'),
    ('Congestión', 'Diagnóstico'),
    ('Saturación', 'Diagnóstico')
])

# --- 2. Definición de las CPDs ---
cpd_virus = TabularCPD('Virus', 2, [[0.7], [0.3]])
cpd_alergia = TabularCPD('Alergia', 2, [[0.8], [0.2]])
cpd_tabaquismo = TabularCPD('Tabaquismo', 2, [[0.75], [0.25]])

cpd_fiebre = TabularCPD('Fiebre', 2, [[0.8, 0.2], [0.2, 0.8]], evidence=['Virus'], evidence_card=[2])
cpd_tos = TabularCPD('Tos', 2, 
    [[0.8, 0.6, 0.2, 0.1],
     [0.2, 0.4, 0.8, 0.9]],
    evidence=['Virus','Tabaquismo'], evidence_card=[2,2])

cpd_congestion = TabularCPD('Congestión', 2,
    [[0.7,0.5,0.2,0.1],
     [0.3,0.5,0.8,0.9]],
    evidence=['Virus','Alergia'], evidence_card=[2,2])

cpd_saturacion = TabularCPD('Saturación', 2, [[0.7,0.1],[0.3,0.9]], evidence=['Fiebre'], evidence_card=[2])

cpd_diagnostico = TabularCPD('Diagnóstico', 2,
    [[0.95,0.8,0.7,0.6,0.4,0.3,0.2,0.05],
     [0.05,0.2,0.3,0.4,0.6,0.7,0.8,0.95]],
    evidence=['Tos','Congestión','Saturación'], evidence_card=[2,2,2])

model.add_cpds(cpd_virus, cpd_alergia, cpd_tabaquismo, cpd_fiebre, cpd_tos, cpd_congestion, cpd_saturacion, cpd_diagnostico)

# --- 3. Inferencias ---
infer = VariableElimination(model)

print("\nP(Diagnóstico | Fiebre=Alta, Tos=Sí):")
print(infer.query(variables=['Diagnóstico'], evidence={'Fiebre':1, 'Tos':1}))

print("\nP(Diagnóstico | Congestión=Sí, Saturación=Baja):")
print(infer.query(variables=['Diagnóstico'], evidence={'Congestión':1, 'Saturación':0}))

print("\nP(Tos | Virus=Sí):")
print(infer.query(variables=['Tos'], evidence={'Virus':1}))

print("\nP(Congestión | Alergia=Sí):")
print(infer.query(variables=['Congestión'], evidence={'Alergia':1}))

print("\nP(Saturación | Fiebre=Alta):")
print(infer.query(variables=['Saturación'], evidence={'Fiebre':0}))
