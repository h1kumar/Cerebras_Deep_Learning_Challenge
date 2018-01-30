import Oil_Spill_Model as osm

## Basic Testing of Relu Functions and their Derivatives
a,_ = osm.relu(0,0,0)
assert a == 0

a   = osm.relu_derivative(1,-1)
assert a == 0

a   = osm.relu_derivative(1,1)
assert a == 1

a,_ = osm.sigmoid([[-1],[0],[1]],[[1,1,1]],[0])
assert a == 0.5

a = osm.sigmoid_derivative(4,0.5)
assert a == 1