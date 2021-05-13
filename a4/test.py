import numpy as np
import skfuzzy as fuzzy
from skfuzzy import control as ctrl

if __name__ == '__main__':

    D = ctrl.Antecedent(np.linspace(0, 10, 21), 'distance')
    D['N'] = fuzzy.trimf(D.universe, [0, 0, 5])
    D['F'] = fuzzy.trimf(D.universe, [0, 5, 10])
    D['VF'] = fuzzy.trimf(D.universe, [5, 10, 10])
    D.view()

    A = ctrl.Antecedent(np.linspace(0, 90, 91), 'angle')
    A['S'] = fuzzy.trimf(A.universe, [0, 0, 45])
    A['M'] = fuzzy.trimf(A.universe, [0, 45, 90])
    A['L'] = fuzzy.trimf(A.universe, [45, 90, 90])
    A.view()

    S = ctrl.Consequent(np.linspace(0, 5, 26), 'speed')
    S['SS'] = fuzzy.trimf(S.universe, [0, 0, 2])
    S['MS'] = fuzzy.trimf(S.universe, [0, 2, 5])
    S['FS'] = fuzzy.trimf(S.universe, [0, 3, 5])
    S['MXS'] = fuzzy.trimf(S.universe, [3, 5, 5])
    S.view()

    ST = ctrl.Consequent(np.linspace(0, 90, 91), 'turn')
    ST['MST'] = fuzzy.trimf(ST.universe, [0, 0, 45])
    ST['SST'] = fuzzy.trimf(ST.universe, [0, 45, 90])
    ST['VST'] = fuzzy.trimf(ST.universe, [45, 90, 90])
    ST.view()

    # rule1 = ctrl.Rule(dis)

    rule1 = ctrl.Rule(D['N'] & A['S'], ST['MST'])
    rule2 = ctrl.Rule(D['N'] & A['M'], ST['SST'])
    rule3 = ctrl.Rule(D['N'] & A['L'], ST['VST'])
    rule4 = ctrl.Rule(D['F'] & A['S'], ST['MST'])
    rule5 = ctrl.Rule(D['F'] & A['M'], ST['SST'])
    rule6 = ctrl.Rule(D['F'] & A['L'], ST['VST'])
    rule7 = ctrl.Rule(D['VF'] & A['S'], ST['MST'])
    rule8 = ctrl.Rule(D['VF'] & A['M'], ST['SST'])
    rule9 = ctrl.Rule(D['VF'] & A['L'], ST['VST'])

    rule10 = ctrl.Rule(D['N'] & A['S'], S['SS'])
    rule11 = ctrl.Rule(D['N'] & A['M'], S['SS'])
    rule12 = ctrl.Rule(D['N'] & A['L'], S['SS'])
    rule13 = ctrl.Rule(D['F'] & A['S'], S['FS'])
    rule14 = ctrl.Rule(D['F'] & A['M'], S['MS'])
    rule15 = ctrl.Rule(D['F'] & A['L'], S['MS'])
    rule16 = ctrl.Rule(D['VF'] & A['S'], S['MXS'])
    rule17 = ctrl.Rule(D['VF'] & A['M'], S['MXS'])
    rule18 = ctrl.Rule(D['VF'] & A['L'], S['FS'])

    car_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9,
                                   rule10, rule11, rule12, rule13, rule14, rule15, rule16,
                                   rule17, rule18])

    motion = ctrl.ControlSystemSimulation(car_ctrl)

    motion.input['distance'] = 9
    motion.input['angle'] = 35

    motion.compute()

    S.view(sim=motion)
    ST.view(sim=motion)

    print(motion.output['speed'])
    print(motion.output['turn'])

    pass