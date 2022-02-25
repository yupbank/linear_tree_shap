# Linear Tree Shap
---
Compute exact shapley value for decision trees in Linear time.

For tree with maximum depth D, and number of leafs L. 
We compute the shapely value is O(LD) time, without the need of extra memory.

# Numerical problems
---
When tree depth exceed 12, with `double` type, there are chances to running into overflow.
which can be mitigated using `long double` data type.


