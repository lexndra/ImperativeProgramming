#ifndef DATASET_H
#define DATASET_H

#include "decision_tree.h"

/* Генерация учебных датасетов для робота. */
Dataset create_robot_classification_dataset(int n_samples);
Dataset create_robot_regression_dataset(int n_samples);

/* Вывод первых строк датасета и освобождение памяти. */
void print_dataset_head(const Dataset *data, TaskType task, int rows);
void free_dataset(Dataset *data);

#endif
