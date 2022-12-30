# Plane-association
This framework is a benchmark for the plane association algorithms in point clouds.
It contains implementations of 2 most popular algorithms, metrics and also loaders for such datasets 
as [TUM](https://vision.in.tum.de/data/datasets/rgbd-dataset) and [ICL NUIM](https://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html).

### How to use
To use this solution you need raw data for association --- only RGBD depth images are supported now.
Also ground truth annotations are required. Solution supports two formats of annotations: colored images, where same planes have same colors in all frames 
and .npy files, where each point from point cloud is associated with plane label which is unique all over the sequence.

### Algorithms
Two most popular algorithms of plane association were implemented.
1. Method based on weighted combination of IoU and plane's normal vector (introduced in [Pop-up SLAM: Semantic monocular plane SLAM for low-texture environments](https://ieeexplore.ieee.org/document/7759204)). In this method IoU, angle between plane normal vectors and distance from the origin are calculated
for each pair of planes from two associated frames. Then these 3 values are combined with weights and pairs with the best score are results of the associations.
2. Method with filtering based on plane's normal vector (introduced in [RGB-D SLAM in Indoor Planar Environments with Multiple Large Dynamic Objects](https://arxiv.org/abs/2203.02882)). In this method  angle between plane normal vectors and distance from the origin are also calculated for each pair of planes, but
after that some of them are filtered out if they aren't close enough according to thresholds. IoU is evaluated only for the remaining pairs and the best score shows result association.

### Metrics
This benchmark can measure quality and performance of each implemented algorithm. Performance is measured by evaluating selected method for each frames' pair for 1000 iterations.
Quality is measured using plane association metrics from [evops-metrics](https://github.com/prime-slam/evops-metrics) library.

### Results
Here you can find result of algorithms comparison on EVOPS dataset. It can be seen that method based on weighted combination of IoU and plane's normal vector shows better quality,
but low performance. However, this problem can be solved by down sampling of the input data. Base down sample solution can be found in `down_sample.py`.
