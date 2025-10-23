
"""
Red Bayesiana Bella (Twilight-verse) usando pomegranate

Nodos:
- W: Weather {Sunny, Cloudy}
- T: Time {Day, Night}
- V: Volturi {Yes, No}
- WA: WolvesAlliance {Strong, Weak}  # independent root (not used downstream)
- VA: VampireActivity {High, Low}    # parents: W, T
- TB: ThreatToBella {High, Low}      # parents: VA, V
- HS: HumanSuspicion {High, Low}     # parents: VA, V
- ED: EdwardDecision {Turn, Protect} # parents: TB, HS
- BS: BellaState {Vampire, Human}    # parent: ED

Se calculan cinco consultas de inferencia en la parte inferior utilizando BayesianNetwork de Pomegranate.

DECLARACIÓN IA:
Se utiliza IA primero para que nos de ejemplos de redes bayesianas, usando la presentación que tenemos en la clase,
utilizando el ejemplo de Rain,Maintenance,Train,Appointment. Despues nos ya con el tema seleccionado, las variables y CPDs construidos 
manualmente, le pedimos que nos explique las librerias pomegranate y DiscreteDistribution para poder a empezar a implementar la red bayesiana de crepusculo.



"""

from collections import OrderedDict
from itertools import product

# -----------------------------
# CPTs 
# -----------------------------

P_W = {'Sunny': 0.3, 'Cloudy': 0.7}
P_T = {'Day': 0.5, 'Night': 0.5}
P_V = {'Yes': 0.2, 'No': 0.8}
P_WA = {'Strong': 0.6, 'Weak': 0.4}  # INDEPENDIENTE

# VA | W,T
P_VA = {
    ('Cloudy','Day'):   {'High':0.8, 'Low':0.2},
    ('Cloudy','Night'): {'High':0.9, 'Low':0.1},
    ('Sunny','Day'):    {'High':0.2, 'Low':0.8},
    ('Sunny','Night'):  {'High':0.6, 'Low':0.4},
}

# TB | VA, V
P_TB = {
    ('High','Yes'): {'High':0.9, 'Low':0.1},
    ('Low', 'Yes'): {'High':0.7, 'Low':0.3},
    ('High','No'):  {'High':0.6, 'Low':0.4},
    ('Low', 'No'):  {'High':0.1, 'Low':0.9},
}

# HS | VA, V
P_HS = {
    ('High','Yes'): {'High':0.8, 'Low':0.2},
    ('Low', 'Yes'): {'High':0.5, 'Low':0.5},
    ('High','No'):  {'High':0.4, 'Low':0.6},
    ('Low', 'No'):  {'High':0.1, 'Low':0.9},
}

# ED | TB, HS
P_ED = {
    ('High','High'): {'Turn':0.7,  'Protect':0.3},
    ('High','Low'):  {'Turn':0.5,  'Protect':0.5},
    ('Low','High'):  {'Turn':0.3,  'Protect':0.7},
    ('Low','Low'):   {'Turn':0.05, 'Protect':0.95},
}

# BS | ED
P_BS = {
    ('Turn',):   {'Vampire':0.95, 'Human':0.05},
    ('Protect',):{'Vampire':0.05, 'Human':0.95},
}

# DOMINIOS
domains = OrderedDict([
    ('W', ['Sunny','Cloudy']),
    ('T', ['Day','Night']),
    ('V', ['Yes','No']),
    ('WA', ['Strong','Weak']),  # independent
    ('VA', ['High','Low']),
    ('TB', ['High','Low']),
    ('HS', ['High','Low']),
    ('ED', ['Turn','Protect']),
    ('BS', ['Vampire','Human']),
])

def joint_prob(assignment):
    """Compute joint probability of a complete assignment using the BN factorization."""
    w, t, v, wa = assignment['W'], assignment['T'], assignment['V'], assignment['WA']
    va, tb, hs, ed, bs = assignment['VA'], assignment['TB'], assignment['HS'], assignment['ED'], assignment['BS']

    p = 1.0
    p *= P_W[w]
    p *= P_T[t]
    p *= P_V[v]
    p *= P_WA[wa]
    p *= P_VA[(w,t)][va]
    p *= P_TB[(va,v)][tb]
    p *= P_HS[(va,v)][hs]
    p *= P_ED[(tb,hs)][ed]
    p *= P_BS[(ed,)][bs]
    return p

def enumerate_ask(query_var, query_val, evidence):
    """Exact inference by enumeration: returns P(query_var=query_val | evidence)."""
    hidden_vars = [X for X in domains.keys() if X not in evidence and X != query_var]

    def total_for(val):
        total = 0.0
        for values in product(*[domains[h] for h in hidden_vars]):
            a = dict(zip(hidden_vars, values))
            a.update(evidence)
            a[query_var] = val
            total += joint_prob(a)
        return total

    num = total_for(query_val)
    den = sum(total_for(v) for v in domains[query_var])
    return num / den

# pomegranate 
_use_pomegranate = True
try:
    from pomegranate import BayesianNetwork, DiscreteDistribution, ConditionalProbabilityTable, Node
except Exception:
    _use_pomegranate = False

def build_with_pomegranate():
    from pomegranate import BayesianNetwork, DiscreteDistribution, ConditionalProbabilityTable, Node

    # RAICES
    W = DiscreteDistribution(P_W)
    T = DiscreteDistribution(P_T)
    V = DiscreteDistribution(P_V)
    WA = DiscreteDistribution(P_WA)

    # CPTs 
    VA_rows = []
    for w in ['Cloudy','Sunny']:   
        for t in ['Day','Night']:
            key = (w,t)
            VA_rows.append([w,t,'High',P_VA[key]['High']])
            VA_rows.append([w,t,'Low', P_VA[key]['Low']])

    VA = ConditionalProbabilityTable(VA_rows, [W, T])

    TB_rows = []
    for va in ['High','Low']:
        for v in ['Yes','No']:
            key = (va,v)
            TB_rows.append([va,v,'High',P_TB[key]['High']])
            TB_rows.append([va,v,'Low', P_TB[key]['Low']])

    TB = ConditionalProbabilityTable(TB_rows, [VA, V])

    HS_rows = []
    for va in ['High','Low']:
        for v in ['Yes','No']:
            key = (va,v)
            HS_rows.append([va,v,'High',P_HS[key]['High']])
            HS_rows.append([va,v,'Low', P_HS[key]['Low']])

    HS = ConditionalProbabilityTable(HS_rows, [VA, V])

    ED_rows = []
    for tb in ['High','Low']:
        for hs in ['High','Low']:
            key = (tb,hs)
            ED_rows.append([tb,hs,'Turn',   P_ED[key]['Turn']])
            ED_rows.append([tb,hs,'Protect',P_ED[key]['Protect']])

    ED = ConditionalProbabilityTable(ED_rows, [TB, HS])

    BS_rows = []
    for ed in ['Turn','Protect']:
        key = (ed,)
        BS_rows.append([ed,'Vampire',P_BS[key]['Vampire']])
        BS_rows.append([ed,'Human',  P_BS[key]['Human']])

    BS = ConditionalProbabilityTable(BS_rows, [ED])

    # Build network
    nodes = {
        'W': Node(W, name='W'),
        'T': Node(T, name='T'),
        'V': Node(V, name='V'),
        'WA': Node(WA, name='WA'),
        'VA': Node(VA, name='VA'),
        'TB': Node(TB, name='TB'),
        'HS': Node(HS, name='HS'),
        'ED': Node(ED, name='ED'),
        'BS': Node(BS, name='BS'),
    }

    bn = BayesianNetwork("Bella BN")
    for n in nodes.values():
        bn.add_node(n)

    # Edges
    bn.add_edge(nodes['W'], nodes['VA'])
    bn.add_edge(nodes['T'], nodes['VA'])
    bn.add_edge(nodes['VA'], nodes['TB'])
    bn.add_edge(nodes['V'], nodes['TB'])
    bn.add_edge(nodes['VA'], nodes['HS'])
    bn.add_edge(nodes['V'], nodes['HS'])
    bn.add_edge(nodes['TB'], nodes['ED'])
    bn.add_edge(nodes['HS'], nodes['ED'])
    bn.add_edge(nodes['ED'], nodes['BS'])

    bn.bake()
    return bn

def query_with_pomegranate(bn, query_var, query_val, evidence):
    ev = {**evidence}  
    dists = bn.predict_proba(ev)
    name_order = [node.name for node in bn.states]
    idx = name_order.index(query_var)
    dist = dists[idx]
    if hasattr(dist, 'parameters'):
        params = dist.parameters[0]  
        return params.get(query_val, 0.0)
    else:
        return 1.0 if evidence.get(query_var) == query_val else 0.0

def run_queries():
    queries = OrderedDict()

    # Q1: PREGUNTA PRINCIPAL
    queries['Q1'] = ('BS','Vampire', {'W':'Cloudy','T':'Day','V':'Yes'})

    # Q2: Probabilidad de que Edward decida convertir dado que es una noche soleada y Volturi está ausente
    queries['Q2'] = ('ED','Turn', {'W':'Sunny','T':'Night','V':'No'})

    # Q3: Probabilidad de que la amenaza sea alta dado que Volturi está presente (marginalizando W,T)
    queries['Q3'] = ('TB','High', {'V':'Yes'})

    # Q4: Probabilidad de que la sospecha humana sea alta dado que es un día soleado y Volturi está ausente
    queries['Q4'] = ('HS','High', {'W':'Sunny','T':'Day','V':'No'})

    # Q5: Probabilidad de que Bella sea vampiro dado que es un día soleado y Volturi está ausente
    queries['Q5'] = ('BS','Vampire', {'W':'Sunny','T':'Day','V':'No'})

    results = OrderedDict()

    if _use_pomegranate:
        bn = build_with_pomegranate()
        for key, (qvar, qval, ev) in queries.items():
            results[key] = query_with_pomegranate(bn, qvar, qval, ev)
        backend = 'pomegranate'
    else:
        def convert_ev(ev):
            return ev
        for key, (qvar, qval, ev) in queries.items():
            results[key] = enumerate_ask(qvar, qval, convert_ev(ev))
        backend = 'exact-enumerator'

    return backend, results

if __name__ == "__main__":
    backend, results = run_queries()
    print(f"Backend used: {backend}\n")
    for k,v in results.items():
        print(f"{k}: {v:.5f}")
