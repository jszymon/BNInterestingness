from Utils import SparseDistr

s = SparseDistr((5,5))
print(s.to_array())
print(s.sample(10))
print()

s2 = SparseDistr((5,5), {(1,1): 1.0})
print(s2.to_array())
print(s2.sample(10))
print()

s3 = SparseDistr((5,5), {(1,1): 0.5, (2,2): 0.5}, prior_factor=0.1)
print(s3.to_array())
print(s3.to_array().sum())
x = s3.sample(1_000_000).tolist()
print(x[:10])
print("[1,1] occurred :", sum(r==[1,1] for r in x), "times")
print("[2,2] occurred :", sum(r==[2,2] for r in x), "times")
