# Riesgo_accidente.py
# Implementación de una red Bayesiana para riesgo de accidente vial
# Requiere: pgmpy >= 0.1.25 (aprox). Instalar con: pip install pgmpy

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination


def build_model():
    model = DiscreteBayesianNetwork([
        ('Hora', 'Velocidad'),
        ('Alcohol', 'Riesgo'),
        ('Clima', 'Riesgo'),
        ('Velocidad', 'Riesgo'),
        ('Riesgo', 'Accidente'),
        ('Accidente', 'Severidad')
    ])

    # ----- CPDs raíz -----
    # P(Alcohol)
    cpd_alcohol = TabularCPD(
        variable='Alcohol', variable_card=2,
        values=[[0.8], [0.2]],
        state_names={'Alcohol': ['No', 'Si']}
    )

    # P(Hora)
    cpd_hora = TabularCPD(
        variable='Hora', variable_card=2,
        values=[[0.6], [0.4]],
        state_names={'Hora': ['Dia', 'Noche']}
    )

    # P(Clima)
    cpd_clima = TabularCPD(
        variable='Clima', variable_card=2,
        values=[[0.7], [0.3]],
        state_names={'Clima': ['Seco', 'Lluvia']}
    )

    # ----- CPDs con padres -----
    # P(Velocidad | Hora)
    # filas = estados de Velocidad en orden ['Normal','Alta']
    # columnas = Hora ['Dia','Noche']
    cpd_vel = TabularCPD(
        variable='Velocidad', variable_card=2,
        values=[
            [0.7, 0.4],  # Velocidad = Normal
            [0.3, 0.6],  # Velocidad = Alta
        ],
        evidence=['Hora'], evidence_card=[2],
        state_names={'Velocidad': ['Normal', 'Alta'], 'Hora': ['Dia', 'Noche']}
    )

    # P(Riesgo | Alcohol, Clima, Velocidad)
    # Orden de evidencias y tarjetas:
    #   Alcohol ['No','Si'] (2), Clima ['Seco','Lluvia'] (2), Velocidad ['Normal','Alta'] (2)
    # Convención de columnas (producto cartesiano en el orden dado):
    # (A, C, V): (No,Seco,Normal), (No,Seco,Alta), (No,Lluvia,Normal), (No,Lluvia,Alta),
    #            (Si,Seco,Normal), (Si,Seco,Alta), (Si,Lluvia,Normal), (Si,Lluvia,Alta)
    cpd_riesgo = TabularCPD(
        variable='Riesgo', variable_card=2,
        values=[
            [0.9, 0.7, 0.6, 0.3, 0.6, 0.3, 0.2, 0.1],  # Riesgo = Bajo
            [0.1, 0.3, 0.4, 0.7, 0.4, 0.7, 0.8, 0.9],  # Riesgo = Alto
        ],
        evidence=['Alcohol', 'Clima', 'Velocidad'],
        evidence_card=[2, 2, 2],
        state_names={
            'Riesgo': ['Bajo', 'Alto'],
            'Alcohol': ['No', 'Si'],
            'Clima': ['Seco', 'Lluvia'],
            'Velocidad': ['Normal', 'Alta'],
        }
    )

    # P(Accidente | Riesgo)
    # columnas = Riesgo ['Bajo','Alto']
    cpd_acc = TabularCPD(
        variable='Accidente', variable_card=2,
        values=[
            [0.95, 0.3],  # Accidente = No
            [0.05, 0.7],  # Accidente = Si
        ],
        evidence=['Riesgo'], evidence_card=[2],
        state_names={'Accidente': ['No', 'Si'], 'Riesgo': ['Bajo', 'Alto']}
    )

    # P(Severidad | Accidente)
    # columnas = Accidente ['No','Si']
    cpd_sev = TabularCPD(
        variable='Severidad', variable_card=2,
        values=[
            [1.0, 0.6],  # Severidad = Leve
            [0.0, 0.4],  # Severidad = Grave
        ],
        evidence=['Accidente'], evidence_card=[2],
        state_names={'Severidad': ['Leve', 'Grave'], 'Accidente': ['No', 'Si']}
    )

    # Agregar CPDs
    model.add_cpds(cpd_alcohol, cpd_hora, cpd_clima, cpd_vel, cpd_riesgo, cpd_acc, cpd_sev)

    # Validar
    model.check_model()
    return model


def run_queries(model):
    infer = VariableElimination(model)

    print("1) P(Severidad | Alcohol=Si, Clima=Lluvia):")
    print(infer.query(['Severidad'], evidence={'Alcohol': 'Si', 'Clima': 'Lluvia'}))

    print("\n2) P(Accidente | Riesgo=Alto):")
    print(infer.query(['Accidente'], evidence={'Riesgo': 'Alto'}))

    print("\n3) P(Riesgo | Alcohol=Si, Velocidad=Alta):")
    print(infer.query(['Riesgo'], evidence={'Alcohol': 'Si', 'Velocidad': 'Alta'}))

    print("\n4) P(Velocidad | Hora=Noche):")
    print(infer.query(['Velocidad'], evidence={'Hora': 'Noche'}))

    print("\n5) P(Accidente | Clima=Lluvia, Alcohol=No):")
    print(infer.query(['Accidente'], evidence={'Clima': 'Lluvia', 'Alcohol': 'No'}))


if __name__ == "__main__":
    model = build_model()
    run_queries(model)
