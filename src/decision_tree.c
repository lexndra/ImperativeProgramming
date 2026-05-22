#include "decision_tree.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define EPS 1e-9

typedef struct {
    int feature_index;
    double threshold;
    double score;
} Split;

/* Рекурсивно строит дерево решений. */
static TreeNode *build_tree(const Dataset *data, const int *indices, int n_indices,
                            TaskType task, TreeParams params, int depth);

static TreeNode *create_node(void) {
    TreeNode *node = (TreeNode *)calloc(1, sizeof(TreeNode));
    node->feature_index = -1;
    return node;
}

static void shuffle_ints(int *array, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int tmp = array[i];
        array[i] = array[j];
        array[j] = tmp;
    }
}

/* Находит самый частый класс в текущем узле. */
static int majority_class(const Dataset *data, const int *indices, int n_indices) {
    int *counts = (int *)calloc((size_t)data->n_classes, sizeof(int));
    for (int i = 0; i < n_indices; i++) {
        int cls = data->y_class[indices[i]];
        if (cls >= 0 && cls < data->n_classes) {
            counts[cls]++;
        }
    }

    int best_class = 0;
    int best_count = counts[0];
    for (int c = 1; c < data->n_classes; c++) {
        if (counts[c] > best_count) {
            best_count = counts[c];
            best_class = c;
        }
    }

    free(counts);
    return best_class;
}

static double mean_value(const Dataset *data, const int *indices, int n_indices) {
    if (n_indices <= 0) return 0.0;
    double sum = 0.0;
    for (int i = 0; i < n_indices; i++) {
        sum += data->y_reg[indices[i]];
    }
    return sum / (double)n_indices;
}

static int is_pure_classification(const Dataset *data, const int *indices, int n_indices) {
    if (n_indices <= 1) return 1;
    int first = data->y_class[indices[0]];
    for (int i = 1; i < n_indices; i++) {
        if (data->y_class[indices[i]] != first) return 0;
    }
    return 1;
}

/* Индекс Джини: чем меньше значение, тем однороднее узел. */
static double gini_from_counts(const int *counts, int n_classes, int total) {
    if (total <= 0) return 0.0;
    double impurity = 1.0;
    for (int c = 0; c < n_classes; c++) {
        double p = (double)counts[c] / (double)total;
        impurity -= p * p;
    }
    return impurity;
}

/* Энтропия используется для критерия Information Gain. */
static double entropy_from_counts(const int *counts, int n_classes, int total) {
    if (total <= 0) return 0.0;
    double entropy = 0.0;
    for (int c = 0; c < n_classes; c++) {
        if (counts[c] > 0) {
            double p = (double)counts[c] / (double)total;
            entropy -= p * (log(p) / log(2.0));
        }
    }
    return entropy;
}

static void class_counts_for_indices(const Dataset *data, const int *indices, int n_indices, int *counts) {
    memset(counts, 0, (size_t)data->n_classes * sizeof(int));
    for (int i = 0; i < n_indices; i++) {
        int cls = data->y_class[indices[i]];
        if (cls >= 0 && cls < data->n_classes) {
            counts[cls]++;
        }
    }
}

static double variance_for_indices(const Dataset *data, const int *indices, int n_indices) {
    if (n_indices <= 0) return 0.0;
    double mean = mean_value(data, indices, n_indices);
    double sum_sq = 0.0;
    for (int i = 0; i < n_indices; i++) {
        double diff = data->y_reg[indices[i]] - mean;
        sum_sq += diff * diff;
    }
    return sum_sq / (double)n_indices;
}

static int compare_doubles(const void *a, const void *b) {
    double x = *(const double *)a;
    double y = *(const double *)b;
    if (x < y) return -1;
    if (x > y) return 1;
    return 0;
}

/* Считает качество разбиения для классификации: Gini gain или Information Gain. */
static double classification_split_score(const Dataset *data, const int *indices, int n_indices,
                                         int feature_index, double threshold,
                                         SplitCriterion criterion, double parent_impurity) {
    int *left_counts = (int *)calloc((size_t)data->n_classes, sizeof(int));
    int *right_counts = (int *)calloc((size_t)data->n_classes, sizeof(int));
    int left_n = 0;
    int right_n = 0;

    for (int i = 0; i < n_indices; i++) {
        int row = indices[i];
        int cls = data->y_class[row];
        if (data->X[row][feature_index] <= threshold) {
            left_counts[cls]++;
            left_n++;
        } else {
            right_counts[cls]++;
            right_n++;
        }
    }

    if (left_n == 0 || right_n == 0) {
        free(left_counts);
        free(right_counts);
        return -1.0;
    }

    double left_impurity;
    double right_impurity;
    if (criterion == CRITERION_INFO_GAIN) {
        left_impurity = entropy_from_counts(left_counts, data->n_classes, left_n);
        right_impurity = entropy_from_counts(right_counts, data->n_classes, right_n);
    } else {
        left_impurity = gini_from_counts(left_counts, data->n_classes, left_n);
        right_impurity = gini_from_counts(right_counts, data->n_classes, right_n);
    }

    double weighted = ((double)left_n / (double)n_indices) * left_impurity +
                      ((double)right_n / (double)n_indices) * right_impurity;
    double score = parent_impurity - weighted;

    free(left_counts);
    free(right_counts);
    return score;
}

/* Считает уменьшение дисперсии для задачи регрессии. */
static double regression_split_score(const Dataset *data, const int *indices, int n_indices,
                                     int feature_index, double threshold, double parent_variance) {
    int left_n = 0;
    int right_n = 0;

    for (int i = 0; i < n_indices; i++) {
        int row = indices[i];
        if (data->X[row][feature_index] <= threshold) left_n++;
        else right_n++;
    }

    if (left_n == 0 || right_n == 0) return -1.0;

    int *left_indices = (int *)malloc((size_t)left_n * sizeof(int));
    int *right_indices = (int *)malloc((size_t)right_n * sizeof(int));
    int l = 0;
    int r = 0;
    for (int i = 0; i < n_indices; i++) {
        int row = indices[i];
        if (data->X[row][feature_index] <= threshold) left_indices[l++] = row;
        else right_indices[r++] = row;
    }

    double left_variance = variance_for_indices(data, left_indices, left_n);
    double right_variance = variance_for_indices(data, right_indices, right_n);
    double weighted = ((double)left_n / (double)n_indices) * left_variance +
                      ((double)right_n / (double)n_indices) * right_variance;

    free(left_indices);
    free(right_indices);
    return parent_variance - weighted;
}

/* Перебирает пороги и случайное подмножество признаков, чтобы выбрать лучший split. */
static Split find_best_split(const Dataset *data, const int *indices, int n_indices,
                             TaskType task, TreeParams params) {
    Split best;
    best.feature_index = -1;
    best.threshold = 0.0;
    best.score = -1.0;

    int n_features = data->n_features;
    int max_features = params.max_features;
    if (max_features <= 0 || max_features > n_features) {
        max_features = n_features;
    }

    int *features = (int *)malloc((size_t)n_features * sizeof(int));
    for (int f = 0; f < n_features; f++) features[f] = f;
    shuffle_ints(features, n_features);

    double parent_impurity = 0.0;
    if (task == TASK_CLASSIFICATION) {
        int *parent_counts = (int *)calloc((size_t)data->n_classes, sizeof(int));
        class_counts_for_indices(data, indices, n_indices, parent_counts);
        if (params.criterion == CRITERION_INFO_GAIN) {
            parent_impurity = entropy_from_counts(parent_counts, data->n_classes, n_indices);
        } else {
            parent_impurity = gini_from_counts(parent_counts, data->n_classes, n_indices);
        }
        free(parent_counts);
    } else {
        parent_impurity = variance_for_indices(data, indices, n_indices);
    }

    for (int selected = 0; selected < max_features; selected++) {
        int feature_index = features[selected];
        double *values = (double *)malloc((size_t)n_indices * sizeof(double));
        for (int i = 0; i < n_indices; i++) {
            values[i] = data->X[indices[i]][feature_index];
        }
        qsort(values, (size_t)n_indices, sizeof(double), compare_doubles);

        for (int i = 1; i < n_indices; i++) {
            if (fabs(values[i] - values[i - 1]) < EPS) continue;
            double threshold = (values[i] + values[i - 1]) / 2.0;
            double score;
            if (task == TASK_CLASSIFICATION) {
                score = classification_split_score(data, indices, n_indices, feature_index,
                                                   threshold, params.criterion, parent_impurity);
            } else {
                score = regression_split_score(data, indices, n_indices, feature_index,
                                               threshold, parent_impurity);
            }

            if (score > best.score) {
                best.score = score;
                best.feature_index = feature_index;
                best.threshold = threshold;
            }
        }

        free(values);
    }

    free(features);
    return best;
}

static TreeNode *create_leaf(const Dataset *data, const int *indices, int n_indices, TaskType task) {
    TreeNode *node = create_node();
    node->is_leaf = 1;
    if (task == TASK_CLASSIFICATION) {
        node->predicted_class = majority_class(data, indices, n_indices);
        node->predicted_value = (double)node->predicted_class;
    } else {
        node->predicted_value = mean_value(data, indices, n_indices);
        node->predicted_class = -1;
    }
    return node;
}

/* Рекурсивно строит дерево решений. */
static TreeNode *build_tree(const Dataset *data, const int *indices, int n_indices,
                            TaskType task, TreeParams params, int depth) {
    if (n_indices <= 0) return NULL;

    if (depth >= params.max_depth || n_indices < params.min_samples_split) {
        return create_leaf(data, indices, n_indices, task);
    }

    if (task == TASK_CLASSIFICATION && is_pure_classification(data, indices, n_indices)) {
        return create_leaf(data, indices, n_indices, task);
    }

    Split split = find_best_split(data, indices, n_indices, task, params);
    if (split.feature_index < 0 || split.score <= EPS) {
        return create_leaf(data, indices, n_indices, task);
    }

    int left_count = 0;
    int right_count = 0;
    for (int i = 0; i < n_indices; i++) {
        int row = indices[i];
        if (data->X[row][split.feature_index] <= split.threshold) left_count++;
        else right_count++;
    }

    if (left_count == 0 || right_count == 0) {
        return create_leaf(data, indices, n_indices, task);
    }

    int *left_indices = (int *)malloc((size_t)left_count * sizeof(int));
    int *right_indices = (int *)malloc((size_t)right_count * sizeof(int));
    int l = 0;
    int r = 0;
    for (int i = 0; i < n_indices; i++) {
        int row = indices[i];
        if (data->X[row][split.feature_index] <= split.threshold) left_indices[l++] = row;
        else right_indices[r++] = row;
    }

    TreeNode *node = create_node();
    node->is_leaf = 0;
    node->feature_index = split.feature_index;
    node->threshold = split.threshold;
    node->left = build_tree(data, left_indices, left_count, task, params, depth + 1);
    node->right = build_tree(data, right_indices, right_count, task, params, depth + 1);

    free(left_indices);
    free(right_indices);
    return node;
}

DecisionTree train_decision_tree(const Dataset *data, const int *indices, int n_indices,
                                 TaskType task, TreeParams params) {
    DecisionTree tree;
    tree.root = build_tree(data, indices, n_indices, task, params, 0);
    tree.params = params;
    tree.task = task;
    tree.n_features = data->n_features;
    tree.n_classes = data->n_classes;
    return tree;
}

/* DFS-обход дерева от корня к листу для получения предсказания. */
static const TreeNode *walk_tree(const TreeNode *node, const double *features) {
    if (!node) return NULL;
    if (node->is_leaf) return node;
    if (features[node->feature_index] <= node->threshold) {
        return walk_tree(node->left, features);
    }
    return walk_tree(node->right, features);
}

int predict_tree_class(const DecisionTree *tree, const double *features) {
    const TreeNode *leaf = walk_tree(tree->root, features);
    if (!leaf) return 0;
    return leaf->predicted_class;
}

double predict_tree_regression(const DecisionTree *tree, const double *features) {
    const TreeNode *leaf = walk_tree(tree->root, features);
    if (!leaf) return 0.0;
    return leaf->predicted_value;
}

static void free_node(TreeNode *node) {
    if (!node) return;
    free_node(node->left);
    free_node(node->right);
    free(node);
}

void free_decision_tree(DecisionTree *tree) {
    if (!tree) return;
    free_node(tree->root);
    tree->root = NULL;
}

/* Обучает ансамбль деревьев методом bagging. */
RandomForest train_random_forest(const Dataset *data, TaskType task, TreeParams params, int n_trees) {
    RandomForest forest;
    forest.trees = (DecisionTree *)calloc((size_t)n_trees, sizeof(DecisionTree));
    forest.n_trees = n_trees;
    forest.task = task;
    forest.n_classes = data->n_classes;

    for (int t = 0; t < n_trees; t++) {
        int *bootstrap_indices = (int *)malloc((size_t)data->n_samples * sizeof(int));
        for (int i = 0; i < data->n_samples; i++) {
            bootstrap_indices[i] = rand() % data->n_samples; /* bootstrap-выборка: случайные строки с возвращением */
        }
        forest.trees[t] = train_decision_tree(data, bootstrap_indices, data->n_samples, task, params);
        free(bootstrap_indices);
    }

    return forest;
}

/* Для классификации используется голосование деревьев. */
int predict_forest_class(const RandomForest *forest, const double *features) {
    if (!forest || !forest->trees || forest->n_trees <= 0) return 0;
    int *votes = (int *)calloc((size_t)forest->n_classes, sizeof(int));

    for (int t = 0; t < forest->n_trees; t++) {
        int cls = predict_tree_class(&forest->trees[t], features);
        if (cls >= 0 && cls < forest->n_classes) {
            votes[cls]++;
        }
    }

    int best_class = 0;
    int best_votes = votes[0];
    for (int c = 1; c < forest->n_classes; c++) {
        if (votes[c] > best_votes) {
            best_votes = votes[c];
            best_class = c;
        }
    }

    free(votes);
    return best_class;
}

/* Для регрессии используется среднее значение предсказаний деревьев. */
double predict_forest_regression(const RandomForest *forest, const double *features) {
    if (!forest || !forest->trees || forest->n_trees <= 0) return 0.0;
    double sum = 0.0;
    for (int t = 0; t < forest->n_trees; t++) {
        sum += predict_tree_regression(&forest->trees[t], features);
    }
    return sum / (double)forest->n_trees;
}

void free_random_forest(RandomForest *forest) {
    if (!forest) return;
    if (forest->trees) {
        for (int t = 0; t < forest->n_trees; t++) {
            free_decision_tree(&forest->trees[t]);
        }
        free(forest->trees);
    }
    forest->trees = NULL;
    forest->n_trees = 0;
    forest->n_classes = 0;
}
