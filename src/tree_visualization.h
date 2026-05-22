#ifndef TREE_VISUALIZATION_H
#define TREE_VISUALIZATION_H

#include "decision_tree.h"

/*
 * Визуализация одного дерева решений.
 * Файл DOT можно открыть вручную или превратить в PNG через Graphviz.
 * Файл PNG создаётся командой dot -Tpng, если Graphviz установлен и доступен в PATH.
 */
int export_decision_tree_to_dot(const DecisionTree *tree, const Dataset *data, const char *dot_filename);
int export_decision_tree_to_png(const DecisionTree *tree, const Dataset *data,
                                const char *dot_filename, const char *png_filename);

#endif
