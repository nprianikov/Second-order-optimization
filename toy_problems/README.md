# Toy Problems Demo

## Rosenbrock

The Rosenbrock function is a non-convex function used as a performance test problem for optimization algorithms introduced by Howard H. Rosenbrock in 1960. It is also known as Rosenbrock's valley or Rosenbrock's banana function.

<!-- Rosenbrock function formula -->
![Rosenbrock function formula](https://wikimedia.org/api/rest_v1/media/math/render/svg/5c41765958e94603760f8efbc2d1bf330b696e9c)

![Rosenbrock SGD vs. HF](gifs/rosenbrock.gif)

![Rosenbrock SGD vs. HF](gifs/rosenbrock2.gif)


## Himmelblau

The Himmelblau function is a multi-modal function used as a performance test problem for optimization algorithms introduced by David B. Himmelblau in 1972. The function is defined by

<!-- Himmelblau function formula -->
![Himmelblau function formula](https://wikimedia.org/api/rest_v1/media/math/render/svg/c58cd364f39ccc8ae66ed6c693954bb44c829c62)

Every 10th iteration of SGD (100 iterations in total).
![Himmelblaue SGD](gifs/Himmelblau_sgd_10.gif)

Maximum of 18 iteration were needed to find all four minima for HF.
![Himmelblaue HF](gifs/Himmelblau_hf_1.gif)