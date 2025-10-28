<h1 align="center">AI Coursework (Search, ML and Deep Learning)</h1>


## Overview
This repository contains my portfolio for **CS 6600 - Artificial Intelligence** coursework. I have presented three main assignments organized in their respective subfolders as follows:

- Search
- Machine Learning(ML)
- Deep Learning Applications

Dr. Sathyanarayanan Aakur ([@saakur](https://github.com/saakur), Assistant Professor - Auburn University - (CS 6600 Course Instructor) is credited for designing the course materials as well as the assignment problems. My contribution comes in the form of preparing the solution notebooks as part of the coursework. The results contain my implementation of the AI tools as well as my own performance analysis, visualization, comparison and insights. 

The Final project of this coursework has its own reposity here: https://github.com/mdmirah/Agent-and-Bullet-Detection-in-Combat-Plane/tree/main.

The final project implements a multi-agent reinforcement learning system for the Combat Plane environment using PettingZoo and Stable-Baselines3. It augments the Combat:Plane environment by adding custom functions for agent and bullet detection to design custom heuristics for offensive, defensive and hybrid (Optional and experimental) agents. 

## Educational Purpose Disclaimer
This repository contains my implementations from CS 6600, completed in Summer 2025 at Auburn University. I've added extensive original analysis, visualizations, and insights beyond course requirements.

**To current students**: These assignments are valuable learning experiences. Please complete them independently before referencing this code.

## Search

**Problem:**

This assignment compares four search algorithms for robot path planning: Breadth-First Search (BFS), Depth-First Search (DFS), and two informed A* variants using Euclidean and Manhattan heuristics. The evaluation, conducted across three increasingly complex test cases, focuses on execution time, path length, and the number of nodes visited.

**Results:**

All algorithms produced the same optimal path length, confirming their ability to find the shortest path. However, they differed in execution time and node expansion. 

<img width="1280" height="720" alt="Search" src="https://github.com/user-attachments/assets/35f0e4e3-7f73-45d7-b0e4-6dd80b011963" />

In the simplest case (Test Case 1), DFS was the fastest at 0.000029 seconds, while BFS was slower. The A* variants had slightly higher runtimes but were more efficient in node visits.

In the more complex Test Cases 2 and 3, A* significantly outperformed the uninformed searches. A* with the Manhattan heuristic was the fastest, completing Test Case 3 in 0.016843 seconds, compared to BFS’s 0.064091 seconds. BFS also visited the most nodes—977—while DFS, A* (Euclidean), and A* (Manhattan) visited 741, 888, and 857 nodes respectively.

Interestingly, although both A* variants visited more nodes than DFS in Test Case 3, they still ran faster, indicating better internal efficiency.

The Euclidean heuristic, based on straight-line distance, and the Manhattan heuristic, based on grid-aligned distances, were both admissible and consistent. While both guided A* effectively, the Manhattan heuristic led to consistently faster and leaner searches, making it more efficient for grid-based planning. The Manhattan heuristic outperformed the Euclidean heuristic because it better aligns with the constraints of the grid-based environment used in the path planning problem. In this setting, the robot can only move in orthogonal directions—up, down, left, or right—making straight-line diagonal movement, as assumed by the Euclidean heuristic, impossible. While both heuristics are admissible and consistent, the Euclidean heuristic underestimates the actual cost more frequently due to its assumption of unrestricted movement, leading to less efficient search behavior. In contrast, the Manhattan heuristic provides a more accurate estimate of the true path cost in a grid, allowing A* to focus its exploration more effectively. This resulted in fewer node expansions and faster performance, making the Manhattan heuristic a better fit for the problem.

In conclusion, A* with the Manhattan heuristic is the most suitable choice for this task. It delivered optimal paths with minimal computational cost. BFS, while correct, scaled poorly with complexity, and DFS, though fast in simple cases, was less reliable. Informed search, particularly with well-chosen heuristics, offers both optimality and scalability, making it ideal for practical path planning.

## Machine Learning(ML)

### Problem 1:

This assignemnt is tasked tasked with developing models to predict customer churn for a subscription-based service. Using the provided dataset, the goal is to build two classification models: one using Logistic Regression and the other using Naive Bayes. The task is to compare their performance, interpret the results, and provide insights into customer churn based on findings. Telco Customer Churn dataset is used, which contains customer information such as demographic details, account features, and whether the customer has churned. The target variable is "Churn," indicating whether a customer has left the service.

**Results:**

Logistic Regression consistently outperformed Naive Bayes in terms of Accuracy, Precision, and F1-Score. Naive Bayes exhibited significantly higher Recall, which indicates it correctly identified more churned customers—but at the cost of lower Precision (i.e., more false positives).

Strategy A (Fill Median) performed better across all metrics and models. Strategy B (Drop Column) resulted in a small decline in most performance metrics, particularly for Logistic Regression.

<img width="1280" height="720" alt="Churn" src="https://github.com/user-attachments/assets/91456fdc-960b-4677-b8c3-f3cf61cd85c9" />

Insights Gained: Logistic Regression strikes a better balance between capturing churned customers and minimizing misclassification. Although Naive Bayes has slightly better ROC-AUC, it sacrifices too much precision. Retaining the TotalCharges feature by imputing missing values, preserved valuable information.

Conclusion: **The best-performing model-strategy pair is: Logistic Regression with Strategy A (Fill Median)**. Performance details for this model are: Accuracy: 0.805, Recall: 0.552 (Lower than Naive Bayes), Precision: 0.659, F1-Score: 0.601, ROC_AUC: 0.72 (Slightly lower than Naive Bayes)

### Problem 2:

In this task, we are assigned to implement a fundamental image compression technique using the k-means clustering algorithm from scratch. The core objective is to develop a custom k-means implementation that processes the provided test_image.png by taking a user-specified number of clusters, k, as input. The algorithm will iteratively group the image's pixel colors into k clusters, with the final cluster centroids representing the most prevalent colors in the image. Once clustering is complete, we will compress the image by replacing every original pixel's color with the value of its nearest centroid, effectively reducing the color palette to just k colors. The final step involves saving and visualizing this newly compressed image to demonstrate the practical application of k-means for dimensionality reduction and efficient image representation.

**Results:**

This problem focused on applying color quantization using k-means clustering to compress an image and evaluate how varying the number of clusters k impacts reconstruction error, runtime, and efficiency. The goal was to identify an optimal value of k that achieves a good trade-off between image quality and computational cost.

<img width="1280" height="720" alt="Aubie" src="https://github.com/user-attachments/assets/e0d84bf4-877e-40c4-89ac-00585156f299" />

The reconstruction error decreases rapidly as the number of clusters ( `k` ) increases, dropping from 1840.97 at `k = 2` to just 21.05 at `k = 256`. However, from visual assessment, beyond `k = 32`, further gains in quality become barely noticeable. Starting around `k = 64`, we see diminishing returns in terms of error reduction.

In terms of runtime, the algorithm remains relatively fast up to `k = 8`, with only a slight increase in processing time. After that, the runtime roughly doubles as `k` doubles, reaching 136.48 seconds at `k = 256`, which may be computationally expensive depending on the application.

Efficiency, defined as the percentage of error reduction per percentage of runtime increase, peaks at `k = 4` with a value of 0.46 when compared to the baseline value for `k = 2`.  
This means that `k = 4` offers the most cost-effective improvement in quality. Beyond this point, efficiency drops sharply, and values higher than `k = 32` yield only marginal improvements in error at a high computational cost.

The elbow method, based on the Within-Cluster Sum of Squares (WCSS), shows a clear inflection point between `k = 16` and `k = 32`, suggesting that this range provides an optimal balance between cluster compactness and model complexity.

**Conclusion:**
- **Best balance (quality vs. runtime):** `k = 32`  
- **Most efficient improvement:** `k = 4`  
- **High-fidelity option:** `k = 64`  
- **Values ≥ 128:** Offer minimal visual improvement but large computational cost.

## Deep Learning Applications

### Problem 1:

**Problem**
This task involved implementing two different deep learning architectures to solve a multi-label text classification problem: predicting movie genres based on short textual descriptions. The core challenge was to build both a Recurrent Neural Network (RNN) and a Long Short-Term Memory (LSTM) model capable of handling the multi-class, multi-label nature of the problem, where a single movie could belong to multiple genres simultaneously from a set of 20 possible classes, using a sigmoid activation function for independent probability estimation for each genre. The project required a comparative analysis of both models' performance using metrics like accuracy, precision, and recall. An additional exploratory component involved feature engineering by incorporating movie titles alongside descriptions into one of the models to investigate any potential performance improvements. What is required to show is the successful implementation of both models, a detailed performance comparison, and an analysis of the impact of adding title information on classification efficacy.

**Results:**

Models Evaluated:
- RNN (Description only)
- LSTM (Description only)
- RNN (Description + Title)

Performance with Description Only: Both RNN and LSTM models showed poor genre prediction except for the genre Drama due to class imbalance. Only Drama was consistently predicted well with Precision: 0.53 (both models), Recall: 1.00 and F1-score: 0.31. All other genres had precision, recall, and F1-score near zero. Validation Accuracy was fairly similar with RNN: ~0.14 and LSTM: ~0.15. The models overfit on predicting Drama due to its high support (Class imbalance) in the dataset and failed to generalize.

Performance with Description + Title (RNN Only): Significant improvement across many genres. Action: Precision ↑ from 0.00 to 0.68, Recall 0.61. Adventure: Precision ↑ to 0.41, Recall 0.30. Drama: Still strong — Precision 0.54, Recall 0.72, F1 0.62. Other genres like Comedy, Crime, Thriller also saw small gains. Validation Accuracy: Improved to ~0.24. F1-score (samples avg): Improved from 0.31 → 0.34. Micro-average F1-score: 0.36 vs 0.31 before.

**The RNN model performed better with Description + Title compared to Description only.**

<img width="1280" height="720" alt="Movie Genre" src="https://github.com/user-attachments/assets/8eb0666f-ff60-43bf-914a-64489ab54928" />

Conclusion: RNN and LSTM models struggle with genre diversity when trained only on descriptions. Most genres are underrepresented, causing poor recall and zero precision. Adding the Title alongside Description dramatically improves the RNN’s ability to distinguish between genres. RNN (Description + Title) is the best-performing model, achieving higher recall and precision across multiple genres and overall better validation accuracy.

## Problem 2:

**Problem**
This task involved building and training a Convolutional Neural Network (CNN) from the ground up to perform multi-class classification on the Intel Image Classification dataset. The goal was to correctly categorize natural images into one of six distinct classes: buildings, forests, glaciers, mountains, seas, and streets. The project utilized 14,000 images for training the model and a separate set of 3,000 images for validation and testing. What is required to show is a successfully trained model, its performance metrics (such as accuracy and loss) on the validation set, and a demonstration of its capability to generalize across the six natural scene categories.

**Results:**

I compared the performance of two convolutional neural network (CNN) models: a 3-layer CNN and a deeper 6-layer CNN. I also trained both models on grayscale and RGB versions of the Intel Image Classification dataset. The goal was to evaluate how input color information (grayscale vs. RGB) and network depth affected classification accuracy and class-wise performance.

<img width="1280" height="720" alt="Grayscale vs RGB" src="https://github.com/user-attachments/assets/66e301e2-67ac-4430-89b3-069732a1c3b9" />

Grascale: For grayscale images, the 3-layer model achieved a test accuracy of 84.00%, while the 6-layer model reached 81.13%. This suggests that the simpler architecture generalized better to the grayscale dataset. The class-wise results show that the 3-layer model had more consistent performance across most classes. In contrast, the deeper model showed slight improvements in some categories like glacier and forest but underperformed in others such as mountain and buildings, indicating potential overfitting or increased complexity not translating into better generalization for grayscale inputs.

RGB: When the same models were trained on RGB images, the 3-layer model again performed slightly better with a test accuracy of 84.13%, while the 6-layer model dropped to 79.17%. Interestingly, while RGB input introduced more visual features and color cues, it did not significantly boost the 3-layer model’s accuracy compared to its grayscale counterpart. The deeper model, however, performed worse on RGB data than on grayscale, suggesting that the additional complexity combined with RGB input may have led to overfitting or difficulty in training efficiently with the available data.

Prediction Comparison: Visual analysis of predictions on two test samples revealed another layer of insight. For the grayscale-trained models, both the 3-layer and 6-layer CNNs correctly predicted the first image as a building, but both failed on the second image, misclassifying it as street instead of building. However, under RGB training, the 6-layer CNN correctly classified the second image as building, while the 3-layer CNN misclassified it. This suggests that although overall performance of the RGB 6-layer model was lower, it potentially benefited from the richer feature representation of RGB in more challenging or ambiguous samples.

Conclusion: The 3-layer CNN remained more stable and accurate across both grayscale and RGB datasets, while the 6-layer model exhibited inconsistencies likely due to overfitting. The RGB 6-layer model showed specific strengths in complex cases, hinting at the trade-off between model depth and input richness that can influence performance in nuanced ways.

## Acknowledgements

- CS 6600 Course Instructor - Dr. Sathyanarayanan Aakur, Assistant Professor - Auburn University.
- Open Source Community - For excellent AI/ML libraries and tools.
- Peer Collaborators - For insightful discussions and code reviews.
- Generative AI including ChatGPT, Deepseek and Co-pilot - For debugging and formatting the codebase. 

## Citation

If you use this repository in your research or find it helpful for your work, please consider citing it using the following BibTeX entry:

```bibtex
@misc{rahman_ai_coursework_2025,
  title = {AI Coursework: Search Algorithms, Machine Learning, and Deep Learning Implementations},
  author = {Rahman, Md Mijanur},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/mdmirah/AI-Coursework-Search.-ML-and-Deep-Learning-}},
}
