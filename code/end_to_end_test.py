import Oil_Spill_Model as osm
import Oil_Spill_Model_Extra_Credit as osmec

## End to End Testing of both the models
a,b,c = osm.oil_spill_model("./data")
assert 95<=a<=100
assert 95<=b<=100
assert 95<=c<=100

a,b,c = osmec.oil_spill_model_extra_credit("./data")
assert 95<=a<=100
assert 95<=b<=100
assert 95<=c<=100