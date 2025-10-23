"""
Modelo Bayesiano del universo de Crepúsculo (versión comentada y humanizada)

Qué modelamos (nodos):
- Weather: Clima (Sunny/Cloudy) influye en actividad vampírica.
- Time: Momento del día (Day/Night) influye en actividad vampírica.
- Volturi: Si los Volturi están presentes (No/Yes), sube la amenaza y la sospecha humana.
- WolvesAlliance: Fortaleza de la alianza de lobos (Weak/Strong). Si es fuerte, baja la amenaza y la sospecha.
- VampireActivity: Actividad vampírica (High/Low) afectada por Weather y Time.
- ThreatToBella: Amenaza a Bella (High/Low) afectada por actividad vampírica, Volturi y lobos.
- HumanSuspicion: Sospecha entre humanos (High/Low) afectada por actividad vampírica, Volturi y lobos.
- EdwardDecision: Decisión de Edward (Turn/Protect) afectada por amenaza y sospecha.
- BellaState: Estado de Bella (Vampire/Human) depende de la decisión de Edward.

Convenciones de estados (en orden):
- Para binarios, el estado 0 corresponde a la primera etiqueta y el estado 1 a la segunda.
    Ej.: Weather: 0=Sunny, 1=Cloudy.

Nota sobre columnas en CPDs condicionales:
- Las columnas se ordenan por el producto cartesiano de los estados de las evidencias
    en el mismo orden en que aparecen en `evidence=[...]`. La última evidencia es la
    que varía más rápido.
"""
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# 1️⃣ Estructura (DAG)
model =  DiscreteBayesianNetwork([
    ('Weather', 'VampireActivity'),
    ('Time', 'VampireActivity'),
    ('Volturi', 'ThreatToBella'),       
    ('VampireActivity', 'ThreatToBella'),
    ('Volturi', 'HumanSuspicion'),
    ('VampireActivity', 'HumanSuspicion'),
    ('WolvesAlliance', 'ThreatToBella'),
    ('WolvesAlliance', 'HumanSuspicion'),
    ('ThreatToBella', 'EdwardDecision'),
    ('HumanSuspicion', 'EdwardDecision'),
    ('EdwardDecision', 'BellaState')
])

# Nodo raíz independiente no conectado explícitamente en el DAG
model.add_node('WolvesAlliance')

# 2️⃣ CPDs (con nombres de estados para mayor legibilidad)

# Raíces (distribuciones previas)
cpd_weather = TabularCPD(
    'Weather', 2, [[0.3], [0.7]],
    state_names={'Weather': ['Sunny', 'Cloudy']}
)
cpd_time = TabularCPD(
    'Time', 2, [[0.5], [0.5]],
    state_names={'Time': ['Day', 'Night']}
)
cpd_volturi = TabularCPD(
    'Volturi', 2, [[0.8], [0.2]],
    state_names={'Volturi': ['No', 'Yes']}
)
cpd_wolves = TabularCPD(
    'WolvesAlliance', 2, [[0.4], [0.6]],
    state_names={'WolvesAlliance': ['Weak', 'Strong']}
)

# VampireActivity (VA | Weather, Time)
cpd_va = TabularCPD(
    'VampireActivity', 2,
    [
        [0.2, 0.6, 0.8, 0.9],  # High
        [0.8, 0.4, 0.2, 0.1]   # Low
    ],
    evidence=['Weather', 'Time'], evidence_card=[2, 2],
    state_names={
        'VampireActivity': ['High', 'Low'],
        'Weather': ['Sunny', 'Cloudy'],
        'Time': ['Day', 'Night']
    }
)

# ThreatToBella (TB | VampireActivity, Volturi, WolvesAlliance)
cpd_tb = TabularCPD(
    'ThreatToBella', 2,
    [
        # Evidence order: ['VampireActivity','Volturi','WolvesAlliance']
        # Columns: (VA=High,V=No,W=Weak), (High,No,Strong), (High,Yes,Weak), (High,Yes,Strong),
        #          (Low,No,Weak), (Low,No,Strong), (Low,Yes,Weak), (Low,Yes,Strong)
        [0.9, 0.7, 0.7, 0.5, 0.6, 0.4, 0.1, 0.05],  # High
        [0.1, 0.3, 0.3, 0.5, 0.4, 0.6, 0.9, 0.95]   # Low
    ],
    evidence=['VampireActivity', 'Volturi', 'WolvesAlliance'], evidence_card=[2, 2, 2],
    state_names={
        'ThreatToBella': ['High', 'Low'],
        'VampireActivity': ['High', 'Low'],
        'Volturi': ['No', 'Yes'],
        'WolvesAlliance': ['Weak', 'Strong']
    }
)

# HumanSuspicion (HS | VampireActivity, Volturi, WolvesAlliance)
cpd_hs = TabularCPD(
    'HumanSuspicion', 2,
    [
        # Evidence order: ['VampireActivity','Volturi','WolvesAlliance']
        # Columns: (VA=High,V=No,W=Weak), (High,No,Strong), (High,Yes,Weak), (High,Yes,Strong),
        #          (Low,No,Weak), (Low,No,Strong), (Low,Yes,Weak), (Low,Yes,Strong)
        [0.8, 0.7, 0.5, 0.4, 0.4, 0.3, 0.1, 0.08],  # High
        [0.2, 0.3, 0.5, 0.6, 0.6, 0.7, 0.9, 0.92]   # Low
    ],
    evidence=['VampireActivity', 'Volturi', 'WolvesAlliance'], evidence_card=[2, 2, 2],
    state_names={
        'HumanSuspicion': ['High', 'Low'],
        'VampireActivity': ['High', 'Low'],
        'Volturi': ['No', 'Yes'],
        'WolvesAlliance': ['Weak', 'Strong']
    }
)

# EdwardDecision (ED | ThreatToBella, HumanSuspicion)
cpd_ed = TabularCPD(
    'EdwardDecision', 2,
    [
        [0.7, 0.5, 0.3, 0.05],  # Turn
        [0.3, 0.5, 0.7, 0.95]   # Protect
    ],
    evidence=['ThreatToBella', 'HumanSuspicion'], evidence_card=[2, 2],
    state_names={
        'EdwardDecision': ['Turn', 'Protect'],
        'ThreatToBella': ['High', 'Low'],
        'HumanSuspicion': ['High', 'Low']
    }
)

# BellaState (BS | EdwardDecision)
cpd_bs = TabularCPD(
    'BellaState', 2,
    [
        [0.95, 0.05],  # Vampire
        [0.05, 0.95]   # Human
    ],
    evidence=['EdwardDecision'], evidence_card=[2],
    state_names={
        'BellaState': ['Vampire', 'Human'],
        'EdwardDecision': ['Turn', 'Protect']
    }
)

# 3️⃣ Agregar CPDs al modelo
model.add_cpds(cpd_weather, cpd_time, cpd_volturi, cpd_wolves,
               cpd_va, cpd_tb, cpd_hs, cpd_ed, cpd_bs)

# 4️⃣ Verificar estructura
assert model.check_model()
print("✅ Modelo correcto y sin ciclos. (Las CPDs incluyen nombres de estados para facilitar la lectura)")

# 5️⃣ Hacer inferencias
infer = VariableElimination(model)

query = infer.query(variables=['BellaState'], evidence={
    'Weather': 'Cloudy',
    'Time': 'Day',
    'Volturi': 'Yes',
    'WolvesAlliance': 'Strong'
})
print(query)
