import Oil_Spill_Model as osm
import numpy as np

## Relu Function
A = np.random.randn(2,3)
W = np.random.randn(3,2)
B = np.random.randn(3,1)
a,_ = osm.relu(A,W,B)
np.testing.assert_array_equal(a,np.maximum(0,np.dot(W,A)+B))

## Relu Derivative Function
A = np.random.randn(2,3)
W = np.random.randn(2,3)
a = osm.relu_derivative(A,W)
np.testing.assert_array_equal(a,np.multiply(A,W>0))

## Sigmoid Function
A = np.random.randn(2,3)
W = np.random.randn(3,2)
B = np.random.randn(3,1)
a,_ = osm.sigmoid(A,W,B)
np.testing.assert_array_equal(a, 1/ (1 + np.exp(-(np.dot(W,A)+B))))

## Sigmoid Derivative Function
A = np.random.randn(2,3)
W = np.random.randn(2,3)
a = osm.sigmoid_derivative(A,W)
np.testing.assert_array_equal(a, np.multiply(A,W*(1-W)))

print ("All Unit tests Passed")