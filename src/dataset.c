#include "dataset.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Безопасное копирование строк для названий признаков и классов. */
static char *copy_string(const char *text) {
    size_t len = strlen(text) + 1;
    char *result = (char *)malloc(len);
    if (result) {
        memcpy(result, text, len);
    }
    return result;
}

/* Генерация случайного вещественного числа в заданном диапазоне. */
static double random_double(double min_value, double max_value) {
    return min_value + (max_value - min_value) * ((double)rand() / (double)RAND_MAX);
}

/* Ограничение значения диапазоном [min_value; max_value]. */
static double clamp(double value, double min_value, double max_value) {
    if (value < min_value) return min_value;
    if (value > max_value) return max_value;
    return value;
}

/* Создание пустого датасета: выделяется память под признаки и целевые значения. */
static Dataset create_empty_dataset(int n_samples, int n_features, int n_classes, int regression) {
    Dataset data;
    data.n_samples = n_samples;
    data.n_features = n_features;
    data.n_classes = n_classes;
    data.X = (double **)calloc((size_t)n_samples, sizeof(double *));
    data.y_class = regression ? NULL : (int *)calloc((size_t)n_samples, sizeof(int));
    data.y_reg = regression ? (double *)calloc((size_t)n_samples, sizeof(double)) : NULL;
    data.feature_names = (char **)calloc((size_t)n_features, sizeof(char *));
    data.class_names = n_classes > 0 ? (char **)calloc((size_t)n_classes, sizeof(char *)) : NULL;

    for (int i = 0; i < n_samples; i++) {
        data.X[i] = (double *)calloc((size_t)n_features, sizeof(double));
    }

    data.feature_names[0] = copy_string("distance_cm");
    data.feature_names[1] = copy_string("speed_mps");
    data.feature_names[2] = copy_string("angle_deg");
    data.feature_names[3] = copy_string("battery_pct");

    return data;
}

/*
 * Датасет для классификации.
 * По сенсорам робота формируется целевой класс: STOP, TURN_LEFT, TURN_RIGHT,
 * FORWARD или SLOW_DOWN. Небольшой шум добавлен, чтобы данные были реалистичнее.
 */
Dataset create_robot_classification_dataset(int n_samples) {
    Dataset data = create_empty_dataset(n_samples, 4, 5, 0);

    data.class_names[0] = copy_string("STOP");
    data.class_names[1] = copy_string("TURN_LEFT");
    data.class_names[2] = copy_string("TURN_RIGHT");
    data.class_names[3] = copy_string("FORWARD");
    data.class_names[4] = copy_string("SLOW_DOWN");

    for (int i = 0; i < n_samples; i++) {
        double distance = random_double(3.0, 100.0);
        double speed = random_double(0.0, 2.2);
        double angle = random_double(-90.0, 90.0);
        double battery = random_double(8.0, 100.0);

        data.X[i][0] = distance;
        data.X[i][1] = speed;
        data.X[i][2] = angle;
        data.X[i][3] = battery;

        if (distance < 14.0) {
            data.y_class[i] = 0;
        } else if (distance < 33.0 && angle > 18.0) {
            data.y_class[i] = 2;
        } else if (distance < 33.0 && angle < -18.0) {
            data.y_class[i] = 1;
        } else if (battery < 20.0 || (distance < 42.0 && speed > 1.1)) {
            data.y_class[i] = 4;
        } else {
            data.y_class[i] = 3;
        }

        /* Шум в 3% случаев имитирует ошибки датчиков или разметки. */
        if (random_double(0.0, 1.0) < 0.03) {
            data.y_class[i] = rand() % data.n_classes;
        }
    }

    return data;
}

/*
 * Датасет для регрессии.
 * Целевое значение — рекомендуемая безопасная скорость робота.
 */
Dataset create_robot_regression_dataset(int n_samples) {
    Dataset data = create_empty_dataset(n_samples, 4, 0, 1);

    for (int i = 0; i < n_samples; i++) {
        double distance = random_double(3.0, 100.0);
        double speed = random_double(0.0, 2.2);
        double angle = random_double(-90.0, 90.0);
        double battery = random_double(8.0, 100.0);

        data.X[i][0] = distance;
        data.X[i][1] = speed;
        data.X[i][2] = angle;
        data.X[i][3] = battery;

        double safe_speed = 0.25 + distance / 55.0;
        safe_speed -= fabs(angle) / 130.0;
        safe_speed -= speed > 1.5 ? 0.25 : 0.0;
        safe_speed -= battery < 20.0 ? 0.35 : 0.0;
        safe_speed += random_double(-0.08, 0.08);
        data.y_reg[i] = clamp(safe_speed, 0.0, 2.2);
    }

    return data;
}

/* Печать первых строк датасета для демонстрации преподавателю. */
void print_dataset_head(const Dataset *data, TaskType task, int rows) {
    if (!data || !data->X) return;
    if (rows > data->n_samples) rows = data->n_samples;

    printf("\nFirst %d dataset rows:\n", rows);
    for (int f = 0; f < data->n_features; f++) {
        printf("%12s ", data->feature_names[f]);
    }
    if (task == TASK_CLASSIFICATION) {
        printf("%12s\n", "command");
    } else {
        printf("%12s\n", "target_speed");
    }

    for (int i = 0; i < rows; i++) {
        for (int f = 0; f < data->n_features; f++) {
            printf("%12.3f ", data->X[i][f]);
        }
        if (task == TASK_CLASSIFICATION) {
            int cls = data->y_class[i];
            printf("%12s\n", data->class_names[cls]);
        } else {
            printf("%12.3f\n", data->y_reg[i]);
        }
    }
}

/* Освобождение всей динамически выделенной памяти датасета. */
void free_dataset(Dataset *data) {
    if (!data) return;

    if (data->X) {
        for (int i = 0; i < data->n_samples; i++) {
            free(data->X[i]);
        }
        free(data->X);
    }
    free(data->y_class);
    free(data->y_reg);

    if (data->feature_names) {
        for (int i = 0; i < data->n_features; i++) {
            free(data->feature_names[i]);
        }
        free(data->feature_names);
    }

    if (data->class_names) {
        for (int i = 0; i < data->n_classes; i++) {
            free(data->class_names[i]);
        }
        free(data->class_names);
    }

    data->X = NULL;
    data->y_class = NULL;
    data->y_reg = NULL;
    data->feature_names = NULL;
    data->class_names = NULL;
    data->n_samples = 0;
    data->n_features = 0;
    data->n_classes = 0;
}
