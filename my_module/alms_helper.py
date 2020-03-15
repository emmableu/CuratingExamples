def get_P():
    '''
    P ∈ R(v×m) has Pij set to Pit (Mj) the probability of model Mj computed on only the left-out point (xi, yi) ∈ Vt,
    '''
    return 0

def get_L():
    '''
    L ∈ R(m×v) has Lij set to the loss of model Mi on the other left-out point (xj , yj ) ∈ Vt.
    '''


def get_s(validation_set_size):
    return [1/validation_set_size]*validation_set_size

