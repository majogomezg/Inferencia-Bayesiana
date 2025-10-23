# Diagnostico_respiratorio.py

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# --- Estructura ---
model = DiscreteBayesianNetwork([
    ('Virus', 'Fiebre'),
    ('Virus', 'Tos'),
    ('Virus', 'Congestion'),
    ('Alergia', 'Congestion'),
    ('Tabaquismo', 'Tos'),
    ('Fiebre', 'Saturacion'),
    ('Tos', 'Diagnostico'),
    ('Congestion', 'Diagnostico'),
    ('Saturacion', 'Diagnostico')
])

# --- CPDs ---
cpd_virus = TabularCPD('Virus', 2, [[0.7], [0.3]], state_names={'Virus': ['No','Si']})
cpd_alergia = TabularCPD('Alergia', 2, [[0.8], [0.2]], state_names={'Alergia': ['No','Si']})
cpd_tabaq = TabularCPD('Tabaquismo', 2, [[0.75], [0.25]], state_names={'Tabaquismo': ['No','Si']})

cpd_fiebre = TabularCPD(
    'Fiebre', 2, [[0.8, 0.2],[0.2, 0.8]],
    evidence=['Virus'], evidence_card=[2],
    state_names={'Fiebre': ['Alta','Normal'], 'Virus': ['No','Si']}
)

cpd_tos = TabularCPD(
    'Tos', 2,
    [[0.8,0.6,0.2,0.1],   # Tos = Si
     [0.2,0.4,0.8,0.9]],  # Tos = No
    evidence=['Virus','Tabaquismo'], evidence_card=[2,2],
    state_names={'Tos':['Si','No'], 'Virus':['No','Si'], 'Tabaquismo':['No','Si']}
)

cpd_cong = TabularCPD(
    'Congestion', 2,
    [[0.7,0.5,0.2,0.1],   # Congestion = Si
     [0.3,0.5,0.8,0.9]],  # Congestion = No
    evidence=['Virus','Alergia'], evidence_card=[2,2],
    state_names={'Congestion':['Si','No'], 'Virus':['No','Si'], 'Alergia':['No','Si']}
)

cpd_satur = TabularCPD(
    'Saturacion', 2,
    [[0.7,0.1], [0.3,0.9]],   # [Baja, Normal] | Fiebre in [Alta, Normal]
    evidence=['Fiebre'], evidence_card=[2],
    state_names={'Saturacion':['Baja','Normal'], 'Fiebre':['Alta','Normal']}
)

cpd_diag = TabularCPD(
    'Diagnostico', 2,
    [[0.95,0.8,0.7,0.6,0.4,0.3,0.2,0.05],   # D=Positivo
     [0.05,0.2,0.3,0.4,0.6,0.7,0.8,0.95]],  # D=Negativo
    evidence=['Tos','Congestion','Saturacion'], evidence_card=[2,2,2],
    state_names={
        'Diagnostico':['Positivo','Negativo'],
        'Tos':['Si','No'],
        'Congestion':['Si','No'],
        'Saturacion':['Baja','Normal']
    }
)

model.add_cpds(cpd_virus, cpd_alergia, cpd_tabaq, cpd_fiebre, cpd_tos, cpd_cong, cpd_satur, cpd_diag)
model.check_model()

infer = VariableElimination(model)

# --- Consultas de ejemplo ---
print("1) P(Diagnostico | Fiebre=Alta, Tos=Si):")
print(infer.query(['Diagnostico'], evidence={'Fiebre':'Alta','Tos':'Si'}))

print("\n2) P(Diagnostico | Congestion=Si, Saturacion=Baja):")
print(infer.query(['Diagnostico'], evidence={'Congestion':'Si','Saturacion':'Baja'}))

print("\n3) P(Tos | Virus=Si):")
print(infer.query(['Tos'], evidence={'Virus':'Si'}))

print("\n4) P(Congestion | Alergia=Si):")
print(infer.query(['Congestion'], evidence={'Alergia':'Si'}))

print("\n5) P(Saturacion | Fiebre=Alta):")
print(infer.query(['Saturacion'], evidence={'Fiebre':'Alta'}))
