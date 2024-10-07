from Utils import SparseDistr

s = SparseDistr((5,5))
print(s.to_array())


s2 = SparseDistr((5,5), {(1,1): 1.0})
print(s2.to_array())

s3 = SparseDistr((5,5), {(1,1): 0.5, (2,2): 0.5}, prior_factor=0.1)
print(s3.to_array())
print(s3.to_array().sum())
