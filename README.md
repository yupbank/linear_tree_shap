# Linear Tree Shap
---
Compute exact Shapley value for decision trees in Linear time.

For a tree with maximum depth D, the number of leaves L. 
We compute the shapely value is O(LD) time, without the need for extra memory.

### Numerical problems
---
- tree depth exceeds 12; there are chances of overflow with `double` data type.
It can be mitigated using `long double` data type.
- tree depth exceeds 16; even with the `long double` type, there are still chances to overflow. 
It can still be mitigated using the `fraction` data type.

### No numerical problems with v2
---
By using the interpolation method, we can have a numerical stable solution based on Chebyshev nodes of the first or second kind.


