import Oil_Spill_Model as osm
import Oil_Spill_Model_Extra_Credit as osmec

## Basic Testing of Relu Functions and their Derivatives
a,_ = osm.relu(0,0,0)
assert a == 0
a   = osm.relu_derivative(1,-1)
assert a == 0
a   = osm.relu_derivative(1,1)
assert a == 1

a,_ = osmec.relu(0,0,0)
assert a == 0
a   = osmec.relu_derivative(1,-1)
assert a == 0
a   = osmec.relu_derivative(1,1)
assert a == 1