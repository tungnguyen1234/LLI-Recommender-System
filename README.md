# Recommender-system-project


For now, we access file path ".../tensor_movieLens" to run the matrix and tensor latent scaling. The output of the algorithm are the MAE and the MSE of the algorithm.


For matrix scaling, we run ```python matrix_movieLens.py``` and we could configure the ```percent``` variable from the function ```matrix_traintest_score``` before running.


For tensor scaling, for now we will configure the  manually inside the file ```.../tensor_movieLens.py```. The third dimension of features would include either age or occupation or both of the user.

- To run the third dimension of age, decomment the function ```tensor_age``` at line 159.
- To run the third dimension of occupation, decomment the function ```tensor_occupation``` at line 161.
- To run the third dimension of both age and occupation, decomment the function ```tensor_age_occup``` at line 163.
- We could configure the ```percent``` variable from the function ```tensor_traintest_score``` before running.

Then we run: 

```python tensor_movieLens.py```
