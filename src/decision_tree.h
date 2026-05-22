#ifndef DECISION_TREE_H
#define DECISION_TREE_H

#include <stddef.h>

#define MAX_CLASS_NAME 64

/* Тип задачи: классификация команд робота или регрессия безопасной скорости. */
typedef enum {
    TASK_CLASSIFICATION = 0,
    TASK_REGRESSION = 1
} TaskType;

/* Критерии выбора лучшего разбиения в узле дерева. */
typedef enum {
    CRITERION_GINI = 0,
    CRITERION_INFO_GAIN = 1,
    CRITERION_MSE = 2
} SplitCriterion;

/* Матрица признаков X и целевые значения для классификации/регрессии. */
typedef struct {
    double **X;
    int *y_class;
    double *y_reg;
    int n_samples;
    int n_features;
    int n_classes;
    char **feature_names;
    char **class_names;
} Dataset;

/* Узел дерева решений. В листе хранится предсказанный класс или числовое значение. */
typedef struct TreeNode {
    int is_leaf;
    int feature_index;
    double threshold;
    int predicted_class;
    double predicted_value;
    struct TreeNode *left;
    struct TreeNode *right;
} TreeNode;

/* Параметры обучения дерева. max_features отвечает за случайную подвыборку признаков. */
typedef struct {
    int max_depth;
    int min_samples_split;
    int max_features;
    SplitCriterion criterion;
} TreeParams;

/* Одно дерево решений. */
typedef struct {
    TreeNode *root;
    TreeParams params;
    TaskType task;
    int n_features;
    int n_classes;
} DecisionTree;

/* Ансамбль деревьев: здесь реализован bagging. */
typedef struct {
    DecisionTree *trees;
    int n_trees;
    TaskType task;
    int n_classes;
} RandomForest;

DecisionTree train_decision_tree(const Dataset *data, const int *indices, int n_indices,
                                 TaskType task, TreeParams params);
int predict_tree_class(const DecisionTree *tree, const double *features);
double predict_tree_regression(const DecisionTree *tree, const double *features);
void free_decision_tree(DecisionTree *tree);

RandomForest train_random_forest(const Dataset *data, TaskType task, TreeParams params, int n_trees);
int predict_forest_class(const RandomForest *forest, const double *features);
double predict_forest_regression(const RandomForest *forest, const double *features);
void free_random_forest(RandomForest *forest);

#endif
