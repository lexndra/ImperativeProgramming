#include "tree_visualization.h"

#include <stdio.h>
#include <stdlib.h>

/*
 * DOT использует кавычки и обратный слеш как специальные символы.
 * Поэтому перед записью подписей узлов эти символы нужно экранировать.
 */
static void write_dot_escaped(FILE *file, const char *text) {
    if (!text) return;

    for (const char *p = text; *p; p++) {
        if (*p == '"' || *p == '\\') {
            fputc('\\', file);
            fputc(*p, file);
        } else if (*p == '\n') {
            fputs("\\n", file);
        } else {
            fputc(*p, file);
        }
    }
}

/* Возвращает название признака по индексу, если оно есть в датасете. */
static const char *feature_name(const Dataset *data, int feature_index) {
    if (!data || !data->feature_names) return NULL;
    if (feature_index < 0 || feature_index >= data->n_features) return NULL;
    return data->feature_names[feature_index];
}

/* Возвращает название класса по индексу, если оно есть в датасете. */
static const char *class_name(const Dataset *data, int class_index) {
    if (!data || !data->class_names) return NULL;
    if (class_index < 0 || class_index >= data->n_classes) return NULL;
    return data->class_names[class_index];
}

/*
 * Рекурсивно записывает один узел дерева в DOT.
 * Внутренние узлы показывают условие разбиения, листья — итоговое предсказание.
 */
static int write_node(FILE *file, const TreeNode *node, const DecisionTree *tree,
                      const Dataset *data, int *counter) {
    if (!node) return -1;

    int current_id = (*counter)++;

    if (node->is_leaf) {
        fprintf(file,
                "    node%d [shape=box, style=\"filled,rounded\", fillcolor=\"#b7f7b7\", label=\"",
                current_id);

        if (tree->task == TASK_CLASSIFICATION) {
            const char *name = class_name(data, node->predicted_class);
            fputs("class = ", file);
            if (name) {
                write_dot_escaped(file, name);
            } else {
                fprintf(file, "%d", node->predicted_class);
            }
        } else {
            fprintf(file, "value = %.3f", node->predicted_value);
        }

        fputs("\"];\n", file);
    } else {
        const char *name = feature_name(data, node->feature_index);

        fprintf(file,
                "    node%d [shape=ellipse, style=\"filled\", fillcolor=\"#fff7cc\", label=\"",
                current_id);

        if (name) {
            write_dot_escaped(file, name);
        } else {
            fprintf(file, "X[%d]", node->feature_index);
        }
        fprintf(file, " <= %.3f", node->threshold);
        fputs("\"];\n", file);
    }

    if (!node->is_leaf) {
        int left_id = write_node(file, node->left, tree, data, counter);
        int right_id = write_node(file, node->right, tree, data, counter);

        if (left_id >= 0) {
            fprintf(file, "    node%d -> node%d [label=\"yes\"];\n", current_id, left_id);
        }
        if (right_id >= 0) {
            fprintf(file, "    node%d -> node%d [label=\"no\"];\n", current_id, right_id);
        }
    }

    return current_id;
}

int export_decision_tree_to_dot(const DecisionTree *tree, const Dataset *data, const char *dot_filename) {
    if (!tree || !tree->root || !dot_filename) {
        return 0;
    }

    FILE *file = fopen(dot_filename, "w");
    if (!file) {
        return 0;
    }

    fprintf(file, "digraph DecisionTree {\n");
    fprintf(file, "    rankdir=TB;\n");
    fprintf(file, "    bgcolor=\"white\";\n");
    fprintf(file, "    node [fontname=\"Arial\", fontsize=10, color=\"#555555\"];\n");
    fprintf(file, "    edge [fontname=\"Arial\", fontsize=9, color=\"#555555\"];\n");

    int counter = 0;
    write_node(file, tree->root, tree, data, &counter);

    fprintf(file, "}\n");
    fclose(file);
    return 1;
}

int export_decision_tree_to_png(const DecisionTree *tree, const Dataset *data,
                                const char *dot_filename, const char *png_filename) {
    if (!export_decision_tree_to_dot(tree, data, dot_filename)) {
        return 0;
    }

    char command[512];
    snprintf(command, sizeof(command), "dot -Tpng \"%s\" -o \"%s\"", dot_filename, png_filename);

    int result = system(command);
    return result == 0;
}
