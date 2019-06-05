class Neuron:
    def __init__(self, basal_initialiser, soma_intitialiser):
        self.basal_potential = basal_initialiser.sample()
        self.soma_potential = soma_intitialiser.sample()


class PyramidalNeuron(Neuron):
    def __init__(self, apical_initialiser, basal_initialiser, soma_intitialiser):
        super(PyramidalNeuron, self).__init__(basal_initialiser, soma_intitialiser)
        self.apical_potential = apical_initialiser


class InterNeuron(Neuron):
    def __init__(self, basal_initialiser, soma_intitialiser):
        super(InterNeuron, self).__init__(basal_initialiser, soma_intitialiser)

